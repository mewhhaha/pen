//! Native front end: windows, rendering, audio, headless verification and packaging.
// The player runtime opens no console window on Windows (an error it stops
// with goes to a crash report and a message box); the development one keeps
// its console.
#![cfg_attr(all(windows, not(feature = "compiler")), windows_subsystem = "windows")]
mod accessibility;
mod app_bundle;
mod asset_graph;
mod asset_lint;
mod atlas;
mod audio;
mod bench;
#[cfg(feature = "compiler")]
mod blot_test;
mod bugreport;
#[cfg(all(test, feature = "compiler"))]
mod codec_round_trip;
mod crash;
mod delta;
mod dev;
mod devlog;
#[cfg(feature = "devtools")]
mod devtools;
mod diverge;
#[cfg(test)]
mod fixtures;
mod fuzz;
mod game;
mod gamepad;
#[cfg(feature = "compiler")]
mod gltf;
mod input;
mod logger;
mod loudness;
mod materials;
mod mods;
mod net;
#[cfg(feature = "compiler")]
mod new;
#[cfg(feature = "compiler")]
mod optimize;
mod opus;
mod packets;
mod patch;
mod pipeline;
#[cfg(feature = "compiler")]
mod plugin_build;
mod presets;
mod program;
mod project;
mod publish;
#[cfg(feature = "compiler")]
mod push;
mod record;
mod recording;
#[cfg(feature = "devtools")]
mod remote;
mod reports;
mod reproducible;
mod review;
mod saves;
mod scene_assets;
mod script;
mod services;
mod session;
mod snapshot;
mod spectate;
mod staging;
mod startup;
#[cfg(feature = "steam")]
mod steam;
mod steam_input;
mod store;
#[cfg(feature = "compiler")]
mod textures;
mod timeline;
mod tunables;
mod visual;
mod web;
mod web_export;
mod workshop;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use blot_host::GuestModule;
use pen_protocol::boundary::read_catalog;
use pen_protocol::{Catalog, Viewport};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::project::{Identity, Project};
use crate::session::{Gpu, Options, Paths, Playback, Session};

static IDENTITY: OnceLock<Identity> = OnceLock::new();

/// The game this process plays, as its checkout's `pen.json` or its
/// bundle's manifest names it: its data directory, executable and title.
/// Until it is known, the executable's name, which a package's is.
pub fn identity() -> &'static Identity {
    IDENTITY.get_or_init(|| {
        let exe = std::env::current_exe().ok();
        Identity::named(&exe.as_deref().and_then(Path::file_stem).map_or("pen".into(), |name| name.to_string_lossy()))
    })
}

/// Where the game keeps saves, preferences and crash reports, unless
/// `--data DIR` says otherwise: the platform data directory's, by its name.
fn data_dir() -> PathBuf {
    dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join(&identity().name)
}

/// pen's caches, in the platform cache directory.
fn cache_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|directory| directory.join("pen"))
}

/// What pen keeps in its cache directory, which `pen clean` deletes:
/// compiled games, compiled plugins, what plugins run and keep, the GPU's
/// compiled pipelines, what asset steps built, and tests' scratch files.
const CACHES: [&str; 7] = ["wasm", "components", "programs", "plugins", "pipelines", "assets", "tmp"];

/// `pen clean`: deletes pen's caches, and nothing else there.
fn clean() -> Result<()> {
    let root = cache_dir().context("this system has no cache directory")?;
    let mut freed = 0;
    for name in CACHES {
        let directory = root.join(name);
        if !directory.exists() {
            continue;
        }
        let bytes = project::size_of(&directory);
        std::fs::remove_dir_all(&directory).with_context(|| format!("deleting {}", directory.display()))?;
        eprintln!("deleted {} ({:.1} MB)", directory.display(), bytes as f64 / 1e6);
        freed += bytes;
    }
    eprintln!("pen clean: {:.1} MB freed", freed as f64 / 1e6);
    Ok(())
}

/// The game's module: `game.cwasm` when the package precompiled it for this
/// runtime, else `game.wasm` compiled now (and cached).
fn load_module(engine: &wasmtime::Engine, game: &Path, cache: bool, measure_systems: bool) -> Result<GuestModule> {
    let path = game.join("game.wasm");
    let wasm = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let cache = if cache { cache_dir().map(|directory| directory.join("wasm")) } else { None };
    let started = Instant::now();
    #[cfg(feature = "compiler")]
    if let Some(path) = std::env::var_os("PEN_TRACE_SYSTEMS") {
        let limit =
            std::env::var("PEN_TRACE_LIMIT").ok().map(|value| value.parse::<u64>()).transpose()?.unwrap_or(1_000_000);
        let trace = blot_host::SystemTrace::create(Path::new(&path), limit)?;
        eprintln!("system trace: {} (up to {limit} events; summary on exit)", Path::new(&path).display());
        return GuestModule::traced(engine, &wasm, cache.as_deref(), trace);
    }
    #[cfg(feature = "compiler")]
    if measure_systems && blot_host::read_abi(&wasm)?.function("main_trace").is_some() {
        return GuestModule::traced(engine, &wasm, cache.as_deref(), blot_host::SystemTrace::live());
    }
    let _ = measure_systems;
    let cwasm = game.join("game.cwasm");
    let precompiled = std::fs::read(&cwasm).ok();
    // The runtime the package says it precompiled game.cwasm for, when not this one.
    let foreign = project::Manifest::read(game)?
        .and_then(|manifest| manifest.runtime)
        .filter(|runtime| runtime != project::RUNTIME);
    let module = match (precompiled, foreign) {
        (Some(_), Some(runtime)) if !blot_host::COMPILER => bail!(
            "{} was precompiled for a pen runtime of {runtime}, and this one is {}: package the game again \
             with `pen build`",
            cwasm.display(),
            project::RUNTIME
        ),
        (Some(_), Some(runtime)) => {
            eprintln!("{} was precompiled for another pen ({runtime}); compiling game.wasm", cwasm.display());
            GuestModule::new(engine, &wasm, cache.as_deref())?
        }
        // SAFETY: game.cwasm ships in the same package as this executable,
        // written by `pen build`.
        (Some(precompiled), None) => unsafe {
            GuestModule::with_precompiled(engine, &wasm, &precompiled, cache.as_deref())?
        },
        (None, _) if !blot_host::COMPILER => {
            bail!(
                "{} is missing, and this runtime has no compiler: it plays games packaged by `pen build`",
                cwasm.display()
            )
        }
        (None, _) => GuestModule::new(engine, &wasm, cache.as_deref())?,
    };
    eprintln!("game module ready in {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
    startup::mark("program loaded");
    Ok(module)
}

/// What the window needs to start a session once it exists.
struct Pending {
    options: Options,
    playback: Playback,
    paths: Paths,
    engine: wasmtime::Engine,
    module: GuestModule,
    catalog: Catalog,
    present_mode: wgpu::PresentMode,
    size: Option<(u32, u32)>,
    software: bool,
    /// Dynamic resolution's frame budget (`--frame-budget`).
    frame_budget: Option<f32>,
    /// The dev tools, in `dev` and `run`.
    #[cfg(feature = "devtools")]
    devtools: Option<devtools::Settings>,
    /// Write the last frame with the dev tools (`--overlay-shot`).
    overlay_shot: Option<PathBuf>,
    mods: Option<mods::Mods>,
}

struct App {
    pending: Option<Pending>,
    window: Option<Arc<Window>>,
    session: Option<Session>,
    /// The game's interface for assistive technology.
    accessibility: Option<accessibility::Accessibility>,
    failure: Option<anyhow::Error>,
    /// Steam, until the session's services take it.
    #[cfg(feature = "steam")]
    steam: Option<steam::Steam>,
}

impl App {
    fn start(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let Some(pending) = self.pending.take() else { return Ok(()) };
        // Shown once its accessibility adapter exists, as platforms need.
        let mut attributes = Window::default_attributes().with_title(identity().title.as_str()).with_visible(false);
        attributes = match pending.size {
            Some((width, height)) => attributes.with_inner_size(PhysicalSize::new(width, height)),
            None => attributes.with_inner_size(LogicalSize::new(960.0, 640.0)),
        };
        let window = Arc::new(event_loop.create_window(attributes)?);
        self.accessibility = Some(accessibility::Accessibility::windowed(event_loop, &window)?);
        window.set_visible(true);
        // A recording's frames keep its size, however the window turns out.
        let pinned = pending.size.filter(|_| pending.options.record.is_some());
        let mut gpu = Gpu::windowed(
            Box::new(event_loop.owned_display_handle()),
            window.clone(),
            pending.present_mode,
            &pending.paths.assets,
            pinned,
            pending.software,
            // Captured frames are read back as 8 bits, so HDR is for play alone.
            std::env::var_os("PEN_HDR").is_some_and(|value| value == "1")
                && pending.options.record.is_none()
                && pending.options.screenshot.is_none()
                && pending.options.script.is_empty(),
        )?;
        gpu.renderer.frame_budget(pending.frame_budget);
        #[cfg(feature = "devtools")]
        let (assets, catalog) = (pending.paths.assets.clone(), pending.catalog.clone());
        #[cfg(feature = "steam")]
        let scene = pending.paths.scene.clone();
        let mut session = Session::start(
            pending.options,
            pending.paths,
            pending.engine,
            pending.module,
            pending.catalog,
            gpu,
            Some(&window),
        )?;
        session.playback(pending.playback, Some(&window))?;
        if let Some(size) = pinned {
            session.input.pin(size);
        }
        // A window reads the player's gamepads; a headless run's are its script's.
        session.input.open_gamepads();
        // The game's achievements, stats and rich presence go through its
        // services; the session pumps Steam's callbacks each frame.
        #[cfg(feature = "steam")]
        session.services.register(Box::new(
            services::steam::SteamService::new(self.steam.take(), session.actions.clone()).with_workshop(scene),
        ));
        #[cfg(feature = "devtools")]
        if let Some(settings) = pending.devtools {
            let devtools = devtools::Devtools::new(Some(&window), &session.gpu, catalog, &assets, settings)?;
            session.set_overlay(Box::new(devtools));
        }
        if let Some(path) = pending.overlay_shot {
            session.overlay_shot_at_end(path);
        }
        if let Some(mods) = pending.mods {
            session.set_mods(mods);
        }
        window.request_redraw();
        self.session = Some(session);
        self.window = Some(window);
        Ok(())
    }

    fn close(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(session) = self.session.take()
            && let Err(error) = session.finish()
        {
            self.failure.get_or_insert(error);
        }
        event_loop.exit();
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: anyhow::Error) {
        self.failure = Some(error);
        self.close(event_loop);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Err(error) = self.start(event_loop) {
            self.fail(event_loop, error);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let (Some(window), Some(accessibility)) = (&self.window, &mut self.accessibility) {
            accessibility.window_event(window, &event);
        }
        let (Some(window), Some(session)) = (self.window.clone(), self.session.as_mut()) else { return };
        match &event {
            WindowEvent::CloseRequested => self.close(event_loop),
            WindowEvent::Resized(size) => session.gpu.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let started = Instant::now();
                let stepped = session.step(Some(&window));
                crash::frame(started.elapsed());
                if let (Ok(_), Some(accessibility)) = (&stepped, &mut self.accessibility) {
                    accessibility.frame(session, Some(&window));
                }
                match stepped {
                    Ok(true) => window.request_redraw(),
                    Ok(false) => self.close(event_loop),
                    Err(error) => self.fail(event_loop, error),
                }
            }
            // The dev tools see events first, and keep some from the game.
            _ => {
                if session.overlay_event(&window, &event) {
                    session.input.window_event(&window, &event);
                }
            }
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let Some(session) = &mut self.session {
            session.input.device_event(&event);
        }
    }
}

fn present_mode() -> Result<wgpu::PresentMode> {
    Ok(match std::env::var("PEN_PRESENT_MODE").ok().as_deref() {
        None | Some("auto-vsync") => wgpu::PresentMode::AutoVsync,
        Some("auto-no-vsync") => wgpu::PresentMode::AutoNoVsync,
        Some("fifo") => wgpu::PresentMode::Fifo,
        Some("fifo-relaxed") => wgpu::PresentMode::FifoRelaxed,
        Some("immediate") => wgpu::PresentMode::Immediate,
        Some("mailbox") => wgpu::PresentMode::Mailbox,
        Some(other) => return Err(anyhow!("invalid PEN_PRESENT_MODE {other}")),
    })
}

/// Whether an environment variable is set to anything but 0.
fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|value| value != "0")
}

/// The packaged game beside the executable (or in its macOS application
/// bundle), if there is one.
fn packaged_game() -> Option<PathBuf> {
    app_bundle::packaged_game(&std::env::current_exe().ok()?)
}

/// A command: its name, the words it takes besides options (`[WORD]`
/// optional, `WORD...` any number), whether it works on a checkout, and
/// what it does.
struct CommandSpec {
    name: &'static str,
    operands: &'static str,
    checkout: bool,
    about: &'static str,
}

/// Where the usage tells of an option.
#[derive(PartialEq)]
enum Place {
    /// Among the options for playing.
    Play,
    /// Under each of its commands.
    Command,
    /// In the usage's own words.
    Prose,
}

/// An option: its name, the value it takes ("" for a flag), the commands it
/// is for ("" plays a packaged game, without a command; none, any command),
/// where the usage tells of it, and what it does.
struct OptionSpec {
    name: &'static str,
    value: &'static str,
    commands: &'static [&'static str],
    place: Place,
    about: &'static str,
}

const COMMANDS: [CommandSpec; 27] = [
    CommandSpec {
        name: "tool",
        operands: "NAME [ARGS...]",
        checkout: true,
        about: "run a checkout-owned Rust tool from pen.json; put its arguments after --",
    },
    CommandSpec {
        name: "dev",
        operands: "",
        checkout: true,
        about: "compile the game's Blot program and play it, reloading the program, shaders and assets as they \
                change with the game's state kept, and rerunning the checkout's asset recipes (pipeline.json) \
                as their sources change",
    },
    CommandSpec {
        name: "web",
        operands: "",
        checkout: true,
        about: "browser development server, reloading as the program and assets change, with the dev tools",
    },
    CommandSpec { name: "run", operands: "", checkout: true, about: "compile once and play" },
    CommandSpec {
        name: "build",
        operands: "",
        checkout: true,
        about: "a shippable folder: the player runtime, its libraries and the game (named as pen.json's \
                `game` says), precompiled for the runtime's platform and a CPU baseline, with the assets its \
                catalog reaches, mesh JSON in binary",
    },
    CommandSpec {
        name: "check",
        operands: "",
        checkout: true,
        about: "compile the game and the dev tools, and lint the assets: files the runtime would refuse or that \
                are missing, unused, oversized or without the mip levels they need (pen.json's assets.lints)",
    },
    CommandSpec {
        name: "assets",
        operands: "",
        checkout: true,
        about: "bring the assets pipeline.json's steps make up to date with their sources: a step runs when its \
                sources changed, unless the asset cache holds what it built from the same sources before, which \
                is copied back",
    },
    CommandSpec {
        name: "codegen",
        operands: "",
        checkout: true,
        about: "generate or check protocol code from the embedded schema",
    },
    CommandSpec {
        name: "export",
        operands: "",
        checkout: true,
        about: "export game.wasm, the scene and the assets for verification",
    },
    CommandSpec {
        name: "test",
        operands: "[FILE]",
        checkout: true,
        about: "golden-image tests (default tests/visual/scenarios.json), and a page reviewing their shots",
    },
    CommandSpec { name: "bench", operands: "", checkout: true, about: "guest time per frame, without rendering" },
    CommandSpec {
        name: "packets",
        operands: "DIR",
        checkout: true,
        about: "each frame's input and render packet, without rendering",
    },
    CommandSpec {
        name: "test-blot",
        operands: "[FILE...]",
        checkout: true,
        about: "the checkout's Blot tests (default tests/*.blot, then tests/traps/ and tests/rejected/): each \
                module's test_ entries, compiled through the checkout's compiler plugin (pen.json) as if at \
                src/test_fixture.blot; those in traps/ must trap at `unreachable`, and a module in rejected/ \
                must not compile, with diagnostics holding its `// rejects: TEXT`",
    },
    CommandSpec {
        name: "gltf",
        operands: "FILE",
        checkout: false,
        about: "import a glTF file (.gltf or .glb) as assets: a mesh JSON for each primitive (skinned ones with \
                their rig and the file's animations as clips), its materials and their images, and \
                NAME.import.json listing them; an asset recipe's work",
    },
    CommandSpec {
        name: "diverge",
        operands: "RECORDING",
        checkout: true,
        about: "play an input recording (.penrec) on this build and on another (--against), without rendering, \
                and say where they first differ: the frame, its packet's fields and the state's lanes, by saved \
                section; fails if they differ (for `git bisect run`)",
    },
    CommandSpec {
        name: "fuzz",
        operands: "",
        checkout: true,
        about: "random input against the game's invariants, without rendering: each run plays steps a seed \
                makes up (keys, chords, clicks, drags, the wheel, console commands, tunables, actions on its \
                interface's nodes) and checks that the game does not trap, its packets are valid and its state \
                finite; a run that breaks one is cut down to the steps that still do, written as a script \
                (--script-file plays one back with the same checks)",
    },
    CommandSpec {
        name: "report",
        operands: "FILE",
        checkout: false,
        about: "what a bug report (.penbug: Ctrl+Shift+B in a window, or a script's report step) holds: its \
                build, system and GPU, the end of its log and its recording of the last 30 s, which `--replay \
                FILE` plays; --out DIR unpacks it",
    },
    CommandSpec {
        name: "new",
        operands: "DIR",
        checkout: false,
        about: "a game to start from in the new directory DIR, named after it: pen.json, a Blot program that \
                turns a cube, its assets, a test and pen's generated bindings; it compiles and runs as it is",
    },
    CommandSpec {
        name: "texture",
        operands: "IMAGE",
        checkout: false,
        about: "a PNG as KTX2 files a GPU samples directly, one per format family (name.bc7.ktx2, \
                name.astc.ktx2, name.etc2.ktx2, name.rgba.ktx2), every mip level worked out; a material \
                names name.ktx2 and the renderer reads the family its device has",
    },
    CommandSpec {
        name: "font",
        operands: "FONT",
        checkout: false,
        about: "a TrueType or OpenType font's multi-channel signed distance field atlas (KTX2, RGBA8 read as \
                data) and its glyphs' metrics in ems (the same name's .json), for text sharp at any size",
    },
    CommandSpec {
        name: "patch",
        operands: "OLD NEW",
        checkout: false,
        about: "a patch from one package version to another (--out FILE): package folders, or versions in the \
                store pen build keeps them in (a version's digest or its start; --store, by default the \
                checkout's .pen/store)",
    },
    CommandSpec {
        name: "apply",
        operands: "PATCH [DIR]",
        checkout: false,
        about: "apply a patch to the package in DIR (by default the one beside this executable): only to the \
                version it was made from, every file checked, the new version built beside the old one and \
                then swapped in",
    },
    CommandSpec {
        name: "loudness",
        operands: "FILE [FILE...]",
        checkout: false,
        about: "each WAV or Ogg Opus clip's loudness (EBU R128, LUFS), true peak (dBTP) and length; with \
                --normalize, each brought to its target loudness as far as its true peak allows: a WAV's \
                samples scaled, an Opus file's output gain set",
    },
    CommandSpec {
        name: "clean",
        operands: "",
        checkout: false,
        about: "delete pen's caches in the platform cache directory: compiled games, plugins and GPU \
                pipelines, and what plugins run and keep",
    },
    CommandSpec {
        name: "serve",
        operands: "",
        checkout: false,
        about: "receive packages that `pen push` sends (on a Steam Deck, say), each into a directory of its \
                name, only the files that changed; it listens on 127.0.0.1:7811 unless --listen says where, \
                and takes pushes with its token only",
    },
    CommandSpec {
        name: "push",
        operands: "ADDRESS",
        checkout: false,
        about: "send a package (default .pen/package) to `pen serve` at ADDRESS (port 7811 unless given): \
                only the files the server lacks, checked by their SHA-256",
    },
    CommandSpec {
        name: "remote",
        operands: "ADDRESS [REQUEST...]",
        checkout: false,
        about: "ask a game run with --devtools-listen (port 7812 unless given) one dev tools request, an \
                operation of protocol/schema.json's DevtoolsOperation and up to four numbers (default \
                frame), and print the answer",
    },
];

/// Playing: a packaged game, `dev` and `run`.
const PLAY: &[&str] = &["", "dev", "run"];
/// Every command that works on a checkout.
const CHECKOUT: &[&str] = &[
    "tool",
    "dev",
    "web",
    "run",
    "build",
    "check",
    "assets",
    "codegen",
    "export",
    "test",
    "bench",
    "packets",
    "test-blot",
    "diverge",
    "fuzz",
];

const OPTIONS: [OptionSpec; 102] = [
    OptionSpec { name: "--help", value: "", commands: &[], place: Place::Prose, about: "this" },
    OptionSpec {
        name: "--out",
        value: "FILE",
        commands: &["texture", "font"],
        place: Place::Command,
        about: "the .ktx2 key to write (default IMAGE's or FONT's name with .ktx2)",
    },
    OptionSpec {
        name: "--linear",
        value: "",
        commands: &["texture"],
        place: Place::Command,
        about: "its texels are data (normals, masks), not sRGB colour",
    },
    OptionSpec {
        name: "--flat",
        value: "",
        commands: &["texture"],
        place: Place::Command,
        about: "the full size alone, no smaller mip levels",
    },
    OptionSpec {
        name: "--em",
        value: "N",
        commands: &["font"],
        place: Place::Command,
        about: "pixels per em the glyphs are drawn at (default 48)",
    },
    OptionSpec {
        name: "--range",
        value: "N",
        commands: &["font"],
        place: Place::Command,
        about: "pixels of distance either side of an edge the field holds (default 4)",
    },
    OptionSpec {
        name: "--width",
        value: "N",
        commands: &["font"],
        place: Place::Command,
        about: "the atlas's width in pixels; its height is the power of two that fits (default 512)",
    },
    OptionSpec {
        name: "--weight",
        value: "N",
        commands: &["font"],
        place: Place::Command,
        about: "a variable font's weight (its wght axis)",
    },
    OptionSpec {
        name: "--chars",
        value: "TEXT",
        commands: &["font"],
        place: Place::Command,
        about: "the characters to draw (default the printable ASCII ones)",
    },
    OptionSpec { name: "--version", value: "", commands: &[], place: Place::Prose, about: "pen's version" },
    OptionSpec { name: "--project", value: "DIR", commands: CHECKOUT, place: Place::Prose, about: "the checkout" },
    OptionSpec {
        name: "--no-pipeline",
        value: "",
        commands: &["dev"],
        place: Place::Command,
        about: "leave the asset recipes be",
    },
    OptionSpec {
        name: "--port",
        value: "N",
        commands: &["web", "test"],
        place: Place::Command,
        about: "listen on port N (default the first free one from 8000 to 8015)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["build"],
        place: Place::Command,
        about: "where (default .pen/package, or .pen/web-export with --web)",
    },
    OptionSpec {
        name: "--runtime",
        value: "EXE",
        commands: &["build"],
        place: Place::Command,
        about: "the runtime to ship (default player/ beside this executable); it must be \
                built from the same pen",
    },
    OptionSpec {
        name: "--no-optimize",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "keep the compiler's Wasm unchanged (build normally runs wasm-opt -O3 before shipping)",
    },
    OptionSpec {
        name: "--target",
        value: "TRIPLE",
        commands: &["build"],
        place: Place::Command,
        about: "the runtime's Rust target, when its executable header does not say",
    },
    OptionSpec {
        name: "--cpu",
        value: "PRESET",
        commands: &["build"],
        place: Place::Command,
        about: "the CPU game.cwasm is compiled for: a Cranelift preset, or native (default x86-64-v2 on x86-64)",
    },
    OptionSpec {
        name: "--bundle-id",
        value: "ID",
        commands: &["build"],
        place: Place::Command,
        about: "a macOS runtime's application bundle identifier (default pen.NAME)",
    },
    OptionSpec {
        name: "--web",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "the Rust browser host and the game as static files instead",
    },
    OptionSpec {
        name: "--steam",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "also write steamcmd's app and depot build scripts for the package in steam/ beside it, from \
                pen.json's \"steam\" section ({\"app\": ID, \"depots\": {\"linux\": ID, \"windows\": ID}, \
                \"branch\": NAME}), and say the steamcmd run that uploads them",
    },
    OptionSpec {
        name: "--itch",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "also say the butler push of the package to itch.io, from pen.json's \"itch\" section \
                ({\"target\": \"user/game\"}): to its platform's channel, as a patch",
    },
    OptionSpec {
        name: "--upload",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "run those uploads: steamcmd as STEAM_BUILD_USER (logged in before), butler with \
                BUTLER_API_KEY; without it, nothing is uploaded",
    },
    OptionSpec {
        name: "--label",
        value: "TEXT",
        commands: &["build"],
        place: Place::Command,
        about: "the build's description on Steam and version on itch.io",
    },
    OptionSpec {
        name: "--zip",
        value: "FILE",
        commands: &["build"],
        place: Place::Command,
        about: "with --web, also the site as a ZIP to upload (itch.io's limits are checked)",
    },
    OptionSpec {
        name: "--check",
        value: "",
        commands: &["codegen"],
        place: Place::Command,
        about: "check the generated code instead of writing it",
    },
    OptionSpec {
        name: "--check",
        value: "",
        commands: &["assets"],
        place: Place::Command,
        about: "build nothing: say which steps' outputs differ from what the cache built from their sources, and \
                fail if any do",
    },
    OptionSpec {
        name: "--only",
        value: "STEP",
        commands: &["assets"],
        place: Place::Command,
        about: "only this step (again for more)",
    },
    OptionSpec {
        name: "--cache",
        value: "DIR",
        commands: &["assets"],
        place: Place::Command,
        about: "the asset cache (default PEN_ASSET_CACHE, else assets/ in pen's cache directory)",
    },
    OptionSpec {
        name: "--rust",
        value: "FILE",
        commands: &["codegen"],
        place: Place::Command,
        about: "where the Rust protocol code goes",
    },
    OptionSpec { name: "--out", value: "DIR", commands: &["export"], place: Place::Command, about: "where (required)" },
    OptionSpec { name: "--out", value: "FILE", commands: &["patch"], place: Place::Command, about: "where (required)" },
    OptionSpec {
        name: "--store",
        value: "DIR",
        commands: &["build", "patch"],
        place: Place::Command,
        about: "the store of package versions (default .pen/store in the checkout)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["gltf"],
        place: Place::Command,
        about: "the asset directory written into (default src/assets)",
    },
    OptionSpec {
        name: "--name",
        value: "NAME",
        commands: &["gltf"],
        place: Place::Command,
        about: "what the assets' names start with (default the file's name)",
    },
    OptionSpec {
        name: "--shader",
        value: "KEY",
        commands: &["gltf"],
        place: Place::Command,
        about: "the shader the materials name (default scene.wgsl)",
    },
    OptionSpec {
        name: "--vat",
        value: "FPS",
        commands: &["gltf"],
        place: Place::Command,
        about: "also bake each skinned primitive's clips into a vertex-animation texture, FPS frames a second, \
                with a material drawing with it",
    },
    OptionSpec {
        name: "--vat-shader",
        value: "KEY",
        commands: &["gltf"],
        place: Place::Command,
        about: "the shader those materials name (default vat.wgsl)",
    },
    OptionSpec {
        name: "--blot",
        value: "DIR",
        commands: &["new"],
        place: Place::Command,
        about: "the Blot checkout (blot2) it builds with (default PEN_BLOT, else a blot2 beside DIR or a \
                directory above it)",
    },
    OptionSpec {
        name: "--update",
        value: "",
        commands: &["test", "test-blot"],
        place: Place::Command,
        about: "write the shots as the golden images",
    },
    OptionSpec {
        name: "--review",
        value: "",
        commands: &["test"],
        place: Place::Command,
        about: "serve the review page every run writes (OUT/review.html: each shot beside its golden image and \
                their difference), with buttons that accept a shot as its golden image, until stopped",
    },
    OptionSpec {
        name: "--last",
        value: "",
        commands: &["test"],
        place: Place::Command,
        about: "with --review, serve the last run's page instead of running (no GPU)",
    },
    OptionSpec {
        name: "--only",
        value: "NAME",
        commands: &["test"],
        place: Place::Command,
        about: "only the scenarios whose names hold NAME (again for more)",
    },
    OptionSpec {
        name: "--golden",
        value: "DIR",
        commands: &["test", "test-blot"],
        place: Place::Command,
        about: "the golden images (default golden/ beside FILE; tests/scenarios/golden for --session)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["test", "test-blot"],
        place: Place::Command,
        about: "where shots and differences go (default .pen/visual; .pen/session-tests for --session)",
    },
    OptionSpec {
        name: "--session",
        value: "",
        commands: &["test-blot"],
        place: Place::Command,
        about: "run Blot driver tests against isolated native headless sessions, sequentially (explicit test files required)",
    },
    OptionSpec {
        name: "--filter",
        value: "TEXT",
        commands: &["test-blot"],
        place: Place::Command,
        about: "only the tests whose names hold TEXT",
    },
    OptionSpec {
        name: "--jobs",
        value: "N",
        commands: &["test-blot"],
        place: Place::Command,
        about: "modules compiled at once, each worker in one compiler session, and tests run at once (default half \
                the cores, at most 8)",
    },
    OptionSpec {
        name: "--runs",
        value: "N",
        commands: &["bench"],
        place: Place::Command,
        about: "run the frames N times, keeping each metric's best (default 1)",
    },
    OptionSpec {
        name: "--json",
        value: "",
        commands: &["bench"],
        place: Place::Command,
        about: "print the results on stdout as JSON, for --compare",
    },
    OptionSpec {
        name: "--compare",
        value: "FILE",
        commands: &["bench"],
        place: Place::Command,
        about: "compare with another build's --json results (again for the best of several), and fail when a \
                metric is slower than its threshold",
    },
    OptionSpec {
        name: "--threshold",
        value: "PCT",
        commands: &["bench"],
        place: Place::Command,
        about: "how much slower every metric may get, in percent (default each its own: the median and mean 10, \
                p95 and snapshots 20, loading 25)",
    },
    OptionSpec {
        name: "--against",
        value: "DIR",
        commands: &["diverge"],
        place: Place::Command,
        about: "the other build: a game bundle (`pen export`'s, or a package's game/), as of the recording's \
                commit",
    },
    OptionSpec {
        name: "--seed",
        value: "N",
        commands: &["fuzz"],
        place: Place::Command,
        about: "the first run's seed; each run after takes the next (default 1)",
    },
    OptionSpec {
        name: "--runs",
        value: "N",
        commands: &["fuzz"],
        place: Place::Command,
        about: "how many (default 10)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["fuzz"],
        place: Place::Command,
        about: "where the scripts of runs that break the game go (default .pen/fuzz)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["report"],
        place: Place::Command,
        about: "unpack its files there (report.txt, log.txt, recording.penrec, state.bin)",
    },
    OptionSpec {
        name: "--deadline",
        value: "SECONDS",
        commands: &["test-blot"],
        place: Place::Command,
        about: "how long one test may run before it fails as stuck (default 600; the time it waits for a core \
                counts)",
    },
    OptionSpec {
        name: "--normalize",
        value: "",
        commands: &["loudness"],
        place: Place::Command,
        about: "bring each clip to its target loudness, rewriting it",
    },
    OptionSpec {
        name: "--lufs",
        value: "N",
        commands: &["loudness"],
        place: Place::Command,
        about: "the target loudness (default -23, EBU R128's, or the targets file's)",
    },
    OptionSpec {
        name: "--ceiling",
        value: "N",
        commands: &["loudness"],
        place: Place::Command,
        about: "the highest true peak, in dBTP, normalizing may take a clip to (default -1)",
    },
    OptionSpec {
        name: "--targets",
        value: "FILE",
        commands: &["loudness"],
        place: Place::Command,
        about: "targets as JSON: {\"target\": LUFS, \"peak\": dBTP, \"sounds\": {\"NAME\": LUFS}}",
    },
    OptionSpec { name: "--mute", value: "", commands: PLAY, place: Place::Play, about: "no sound" },
    OptionSpec {
        name: "--audio-buffer",
        value: "N",
        commands: PLAY,
        place: Place::Play,
        about: "the audio device's buffer, N samples within what it takes (default the system's; smaller plays \
                sooner, at more risk of gaps)",
    },
    OptionSpec {
        name: "--headless",
        value: "",
        commands: PLAY,
        place: Place::Play,
        about: "no window; draw offscreen (steady clock, 60 fps unless --fps)",
    },
    OptionSpec {
        name: "--size",
        value: "WxH",
        commands: PLAY,
        place: Place::Play,
        about: "the window's or the offscreen image's size in pixels (headless default 1920x1280, where \
                script coordinates match the browser's)",
    },
    OptionSpec {
        name: "--fps",
        value: "N",
        commands: PLAY,
        place: Place::Play,
        about: "steady game clock: N frames per game second",
    },
    OptionSpec {
        name: "--frames",
        value: "N",
        commands: &["", "dev", "run", "bench", "packets", "fuzz"],
        place: Place::Play,
        about: "stop after N frames (bench: default 600; packets: default 120, or as long as the script lasts; \
                fuzz: each run's, default 600)",
    },
    OptionSpec {
        name: "--screenshot",
        value: "PNG",
        commands: PLAY,
        place: Place::Play,
        about: "write the last frame",
    },
    OptionSpec {
        name: "--shots",
        value: "DIR",
        commands: PLAY,
        place: Place::Play,
        about: "where shot=NAME steps write NAME.png (default shots)",
    },
    OptionSpec {
        name: "--record",
        value: "VIDEO",
        commands: PLAY,
        place: Place::Play,
        about: "record every frame (.mp4, .mkv or .webm, with sound unless --mute)",
    },
    OptionSpec {
        name: "--script",
        value: "STEPS",
        commands: &["", "dev", "run", "bench", "packets", "fuzz"],
        place: Place::Play,
        about: "scripted input: wait=N key=K down=K up=K type=T move=X,Y glide=X,Y,FRAMES click=X,Y press=B release=B \
                drag=X,Y,X,Y wheel=D[,X,Y] shot=NAME save=F load=F quit; ui=ID,ACTION[,VALUE] (an assistive \
                technology's action on an interface node: focus, blur, activate, increment, decrement, \
                set_value, expand or collapse); tune=NAME=VALUE (a live-tunable constant of the checkout's \
                src/tunables.json); overlay=/COMMAND[,ARG...] (a line typed into the dev tools' console); \
                pad=connect|disconnect|BUTTON[,VALUE]|AXIS,VALUE (a gamepad); touch=ID,X,Y lift=ID (a touch \
                screen); motion=DX,DY (the locked mouse); compose=TEXT[,CURSOR] commit=TEXT (an input method); \
                overlay-shot=NAME (a shot with the dev tools over it); \
                glide eases from the last scripted pointer position (initially 0,0) over positive FRAMES, \
                then continues; move is instant",
    },
    OptionSpec {
        name: "--script-file",
        value: "FILE",
        commands: &["", "dev", "run", "bench", "packets", "fuzz"],
        place: Place::Play,
        about: "the same, from a file (lines starting with # are comments)",
    },
    OptionSpec {
        name: "--data",
        value: "DIR",
        commands: PLAY,
        place: Place::Play,
        about: "keep saves, preferences and crash reports in DIR instead of the game's data directory",
    },
    OptionSpec {
        name: "--settle",
        value: "",
        commands: PLAY,
        place: Place::Play,
        about: "draw the first frame until every asset has loaded before the clock starts (default for \
                --headless and --record)",
    },
    OptionSpec { name: "--no-settle", value: "", commands: PLAY, place: Place::Play, about: "start the clock at once" },
    OptionSpec {
        name: "--mod",
        value: "DIR",
        commands: &["", "dev", "run", "packets"],
        place: Place::Play,
        about: "a mod beside the game, again for more: DIR's mod.json and mod.wasm, run in a sandbox with the \
                capabilities its manifest asks for (docs/mods.md)",
    },
    OptionSpec {
        name: "--game",
        value: "DIR",
        commands: &["", "build", "test", "bench", "packets", "diverge", "fuzz"],
        place: Place::Play,
        about: "a game bundle (`pen export`'s, a package's game/, or one built in another language) instead of \
                the checkout's",
    },
    OptionSpec {
        name: "--assets",
        value: "DIR",
        commands: &["", "test", "bench", "packets", "diverge", "fuzz"],
        place: Place::Play,
        about: "the bundle's assets from elsewhere",
    },
    OptionSpec {
        name: "--scene",
        value: "FILE",
        commands: &["", "test", "bench", "packets", "diverge", "fuzz"],
        place: Place::Play,
        about: "the bundle's scene from elsewhere, which the editor saves to",
    },
    OptionSpec {
        name: "--frame-budget",
        value: "MS",
        commands: PLAY,
        place: Place::Play,
        about: "dynamic resolution: lower the scene's render scale (to half at most) while the GPU takes \
                longer than MS milliseconds a frame, and raise it again when it takes well under",
    },
    OptionSpec {
        name: "--software",
        value: "",
        commands: &["", "dev", "run", "test", "test-blot"],
        place: Place::Play,
        about: "draw on a software GPU adapter (lavapipe, WARP), as PEN_SOFTWARE=1 does",
    },
    OptionSpec {
        name: "--record-input",
        value: "FILE",
        commands: PLAY,
        place: Place::Play,
        about: "record the session's input (a .penrec) to replay it exactly",
    },
    OptionSpec {
        name: "--replay",
        value: "FILE",
        commands: PLAY,
        place: Place::Play,
        about: "play recorded input instead of the player's: the same frames, bit for bit (headless, the run \
                ends with it; in a window, it pauses there)",
    },
    OptionSpec {
        name: "--history",
        value: "",
        commands: PLAY,
        place: Place::Play,
        about: "keep the last 30 s to rewind, replay and report (always in a window, dev and replays)",
    },
    OptionSpec {
        name: "--overlap",
        value: "",
        commands: PLAY,
        place: Place::Play,
        about: "compute the next frame while this one draws (one frame more input latency)",
    },
    OptionSpec {
        name: "--trace",
        value: "FILE",
        commands: PLAY,
        place: Place::Play,
        about: "a line per frame: its number, and its input's and packet's hashes",
    },
    OptionSpec {
        name: "--broadcast",
        value: "ADDRESS",
        commands: PLAY,
        place: Place::Play,
        about: "stream the final frames to viewers on ADDRESS (HOST:PORT), over WebSockets",
    },
    OptionSpec {
        name: "--spectate",
        value: "ADDRESS",
        commands: PLAY,
        place: Place::Play,
        about: "watch the stream at ADDRESS (HOST:PORT or ws://...) as it comes, as a replay plays",
    },
    OptionSpec {
        name: "--net",
        value: "PLAYER:ADDRESS,...",
        commands: PLAY,
        place: Place::Play,
        about: "play as PLAYER (from 0) of a net session, every player's UDP address in order (docs/netplay.md)",
    },
    OptionSpec {
        name: "--overlay",
        value: "",
        commands: &["dev", "run"],
        place: Place::Play,
        about: "open the dev tools' console from the start (or PEN_OVERLAY=1)",
    },
    OptionSpec {
        name: "--overlay-shot",
        value: "PNG",
        commands: &["dev", "run"],
        place: Place::Play,
        about: "write the last frame with the dev tools over it (headless too)",
    },
    OptionSpec {
        name: "--devtools-listen",
        value: "ADDRESS",
        commands: PLAY,
        place: Place::Play,
        about: "answer remote dev tools (`pen remote`) at ADDRESS, such as 127.0.0.1:7812",
    },
    OptionSpec {
        name: "--listen",
        value: "ADDRESS",
        commands: &["serve"],
        place: Place::Command,
        about: "where to listen (default 127.0.0.1:7811; 0.0.0.0:7811 takes pushes from the network)",
    },
    OptionSpec {
        name: "--dir",
        value: "DIR",
        commands: &["serve"],
        place: Place::Command,
        about: "where packages go (default the data directory's pushed/)",
    },
    OptionSpec {
        name: "--token",
        value: "TEXT",
        commands: &["serve", "push"],
        place: Place::Command,
        about: "the server's token, which a push must give (or PEN_PUSH_TOKEN; serve makes one up and says it)",
    },
    OptionSpec {
        name: "--allow-run",
        value: "",
        commands: &["serve"],
        place: Place::Command,
        about: "start what a push asks to (after stopping what it started before)",
    },
    OptionSpec {
        name: "--package",
        value: "DIR",
        commands: &["push"],
        place: Place::Command,
        about: "the package to send (default .pen/package)",
    },
    OptionSpec {
        name: "--name",
        value: "NAME",
        commands: &["push"],
        place: Place::Command,
        about: "its name on the server (default the game's, from the package's game/game.json)",
    },
    OptionSpec {
        name: "--run",
        value: "FILE",
        commands: &["push"],
        place: Place::Command,
        about: "start this file of the package there once it is in place (the server must --allow-run)",
    },
    OptionSpec {
        name: "--text",
        value: "TEXT",
        commands: &["remote"],
        place: Place::Command,
        about: "the request's text (a file's path, a preset's name)",
    },
    OptionSpec {
        name: "--follow",
        value: "",
        commands: &["remote"],
        place: Place::Command,
        about: "with log: print the log as it grows",
    },
];

/// Every variable of the environment pen reads, and what it does.
const ENVIRONMENT: [(&str, &str); 38] = [
    ("PEN_PRESENT_MODE=MODE", "auto-vsync (the default), auto-no-vsync, fifo, fifo-relaxed, immediate or mailbox"),
    ("PEN_SOFTWARE=1", "as --software"),
    ("PEN_HDR=1", "HDR output where the display and surface can (not while recording or taking shots)"),
    ("PEN_OVERLAY=1", "as --overlay"),
    (
        "PEN_TRACE_SYSTEMS=FILE",
        "capture per-system CPU time and allocation traffic to Chrome/Perfetto JSON and a .summary.json (native development runtime)",
    ),
    ("PEN_TRACE_LIMIT=N", "maximum timeline events in a system trace (default 1000000); summaries keep counting"),
    ("PEN_SMOKE_FRAMES=N", "as --frames N"),
    ("PEN_AUDIO_BUFFER=N", "as --audio-buffer N"),
    (
        "PEN_PROFILE_FRAMES=N",
        "after PEN_PROFILE_WARMUP=N frames (default 90), time N frames' phases, say each one's median and stop",
    ),
    (
        "PEN_FRAME_TIMES=1",
        "a line on stderr for each frame, with its phases' times, and the renderer's pipeline compile times",
    ),
    (
        "RUST_LOG=LEVEL",
        "more of the libraries' log on stderr: debug, or per crate (pen_render=debug); warnings always show",
    ),
    (
        "PEN_STARTUP=1",
        "when each phase of starting is reached: the program loaded, the GPU ready, the game's first frame, the \
         first frame drawn whole and settling",
    ),
    ("PEN_GPU_BACKTRACE=1", "where a GPU validation error came from"),
    ("PEN_SERVICES_LOG=1", "a line for each request the game makes of its services"),
    (
        "PEN_A11Y_DUMP=FILE",
        "the game's interface as assistive technology sees it, written to FILE whenever it changes (headless too)",
    ),
    ("PEN_A11Y_ATSPI=1", "headless, the game's interface on the AT-SPI bus, at the game's pace (Linux; tests/atspi)"),
    (
        "PEN_A11Y_AUDIT=FILE",
        "the accessibility audit of the game's interface and its pixels, written to FILE whenever it changes",
    ),
    ("PEN_LOCALE=TAG", "the language the game's string tables show in (default the system's locale)"),
    (
        "PEN_COLOR_FILTER=NAME[:STRENGTH]",
        "a colour filter for this run: protanopia, deuteranopia, tritanopia or achromatopsia; simulate-NAME shows what a player with it sees",
    ),
    ("PEN_TEXT_SCALE=N", "the engine's text scale for this run, 1 to 2"),
    ("PEN_REDUCED_MOTION=1", "the engine's reduced motion for this run"),
    ("PEN_RECORD_PCM=FILE", "a recording's sound as mixed, before it is encoded (f32 stereo at 48 kHz)"),
    ("PEN_STEAM=1", "Steam outside a package (pen built with Steam)"),
    ("PEN_STEAM_DEBUG=STEPS", "achievement and stat steps to run against Steam (see steam.rs)"),
    ("PEN_STEAM_APP_ID=ID", "relaunch through Steam when started outside it"),
    ("STEAM_BUILD_USER=NAME", "build --steam --upload: the Steamworks account steamcmd logged in as before"),
    ("PEN_STEAMCMD=EXE, PEN_BUTLER=EXE", "build --upload: steamcmd and butler, if not on PATH"),
    ("BUTLER_API_KEY=KEY", "build --itch --upload: butler's itch.io API key"),
    (
        "PEN_GRAPHICS_TIER=TIER",
        "the graphics preset for a player without stored preferences: auto (from the GPU, and in headless runs \
         too), software, handheld, low, medium or high",
    ),
    ("PEN_CRASH_TEST=N", "panic at frame N, to try crash reports"),
    ("PEN_REPORTS=off", "send no crash or performance reports, whatever the player agreed to"),
    ("PEN_PLUGINS=DIR", "where pen's own plugins are (default plugins/ beside pen)"),
    (
        "PEN_ASSET_CACHE=DIR",
        "what asset steps built, by their sources (default assets/ in pen's cache directory); share one between \
         checkouts and CI runs",
    ),
    (
        "PEN_WEB_HOST=DIR",
        "a built browser host, for web and build --web (default web/ beside pen, else built from this repository)",
    ),
    (
        "BLOT_INLINING=0, =1",
        "Cranelift's inlining: =0 turns it off where it is on (run, build, a package), =1 on where it is off \
         (dev and the dev tools)",
    ),
    ("PEN_BUILD=TEXT", "while building pen: a label for its crash reports"),
    ("PEN_PUSH_TOKEN=TEXT", "as --token, for pen serve and pen push"),
    (
        "SOURCE_DATE_EPOCH=SECONDS",
        "build: every file and directory of the package takes this modification time, so that an archive \
         of it is reproducible too",
    ),
];

/// A line of the usage: `head`, then `about` from column 22, wrapped at 100.
fn entry(text: &mut String, head: &str, about: &str) {
    const COLUMN: usize = 22;
    const WIDTH: usize = 100;
    let mut line = head.to_owned();
    if line.chars().count() >= COLUMN - 1 {
        text.push_str(&line);
        text.push('\n');
        line.clear();
    }
    for word in about.split_whitespace() {
        let width = line.chars().count();
        if width > COLUMN && width + 1 + word.chars().count() > WIDTH {
            text.push_str(&line);
            text.push('\n');
            line.clear();
        }
        let width = line.chars().count();
        line.push_str(&" ".repeat(COLUMN.saturating_sub(width).max(1)));
        line.push_str(word);
    }
    text.push_str(&line);
    text.push('\n');
}

/// `pen` for no command, else `pen COMMAND`.
fn invocation(command: &str) -> String {
    if command.is_empty() { "pen".into() } else { format!("pen {command}") }
}

/// Names in prose: `a`, `a and b`, `a, b and c`.
fn listed(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

fn usage() -> String {
    let mut text = String::from(
        "usage: pen [COMMAND] [OPTIONS]  (pen --help: this; pen --version)\n\n\
         without a command, pen plays a packaged game: game/ beside the executable (a `pen build` package),\n\
         or --game DIR.\n\n\
         commands, in a checkout (a directory with pen.json and src/main.blot; or --project DIR):\n",
    );
    let command = |text: &mut String, spec: &CommandSpec| {
        let head = format!("  {} {}", spec.name, spec.operands);
        entry(text, head.trim_end(), spec.about);
        for option in
            OPTIONS.iter().filter(|option| option.place == Place::Command && option.commands.contains(&spec.name))
        {
            entry(text, format!("    {} {}", option.name, option.value).trim_end(), option.about);
        }
    };
    for spec in COMMANDS.iter().filter(|spec| spec.checkout) {
        command(&mut text, spec);
    }
    text.push_str("\nand anywhere:\n");
    for spec in COMMANDS.iter().filter(|spec| !spec.checkout) {
        command(&mut text, spec);
    }
    text.push_str("\noptions for playing (pen, dev and run):\n");
    for option in OPTIONS.iter().filter(|option| option.place == Place::Play) {
        let named =
            |commands: Vec<&&str>| listed(&commands.into_iter().map(|command| invocation(command)).collect::<Vec<_>>());
        let extra: Vec<&&str> = option.commands.iter().filter(|command| !PLAY.contains(command)).collect();
        let missing = PLAY.iter().any(|command| !option.commands.contains(command));
        let about = match (extra.is_empty(), missing) {
            (true, false) => option.about.to_owned(),
            (false, false) => format!("{}; also {}", option.about, named(extra)),
            (true, true) => format!("{}; {} only", option.about, named(option.commands.iter().collect())),
            (false, true) => format!("{}; for {}", option.about, named(option.commands.iter().collect())),
        };
        entry(&mut text, format!("  {} {}", option.name, option.value).trim_end(), &about);
    }
    text.push_str(
        "\ntime travel: Ctrl+Shift+P pauses and plays, Ctrl+Shift+period steps a frame, Ctrl+Shift+comma\n\
         steps back, Ctrl+Shift+R replays the last 5 s (after a reload, on the new build); in scripts,\n\
         pause resume step[=N] rewind=N replay=SECONDS.\n\n\
         bug reports: Ctrl+Shift+B in a window (a script's report[=FILE]) writes the last 30 s of input,\n\
         the state, the log and the system to a .penbug in reports/ of the data directory (pen report).\n\n\
         dev tools (dev and run): the checkout's own program for them (devtools/main.blot) runs beside the\n\
         game and draws over it: its console on the backquote key (/help lists what it does), and its\n\
         diagnostics; a checkout without one runs without dev tools.\n\n\
         environment:\n",
    );
    for (name, about) in ENVIRONMENT {
        entry(&mut text, &format!("  {name}"), about);
    }
    text.push_str(
        "\nthe game's name (its data directory, and its executable in a package) and title come from the\n\
         checkout's pen.json, {\"game\": {\"name\": \"...\", \"title\": \"...\"}}, else its directory's name; a\n\
         package keeps them in game/game.json. A panic, or an error that stops a package's runtime, writes\n\
         a crash report to crashes/ in the data directory (or --data DIR).",
    );
    text
}

/// How many edits make `a` into `b`.
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, a) in a.chars().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, b) in b.iter().enumerate() {
            let replaced = previous + usize::from(a != *b);
            previous = row[j + 1];
            row[j + 1] = replaced.min(previous + 1).min(row[j] + 1);
        }
    }
    row[b.len()]
}

/// ` (did you mean X?)` for the nearest of `names` to `word`, if one is near.
fn suggestion<'a>(word: &str, names: impl Iterator<Item = &'a str>) -> String {
    names
        .map(|name| (distance(word, name), name))
        .filter(|(distance, _)| *distance <= 2)
        .min()
        .map(|(_, name)| format!(" (did you mean {name}?)"))
        .unwrap_or_default()
}

/// The command line, parsed once: a command, its operands and its options,
/// each checked against [`COMMANDS`] and [`OPTIONS`].
#[derive(Debug, Default)]
struct Args {
    /// The command's name; "" plays a packaged game.
    command: &'static str,
    operands: Vec<String>,
    /// The options given, in order, with their values ("" for a flag).
    options: Vec<(&'static str, String)>,
    /// The steps of `--script` and `--script-file`.
    script: Vec<script::Step>,
}

impl Args {
    fn parse(words: impl IntoIterator<Item = String>) -> Result<Args> {
        let mut words = words.into_iter().peekable();
        let mut args = Args::default();
        if let Some(word) = words.next_if(|word| !word.starts_with('-')) {
            let Some(spec) = COMMANDS.iter().find(|spec| spec.name == word) else {
                let names: Vec<String> = COMMANDS.iter().map(|spec| spec.name.to_owned()).collect();
                bail!(
                    "unknown command {word}{}; the commands are {} (pen --help)",
                    suggestion(&word, COMMANDS.iter().map(|spec| spec.name)),
                    listed(&names)
                );
            };
            args.command = spec.name;
        }
        while let Some(word) = words.next() {
            if args.command == "tool" && word == "--" {
                args.operands.extend(words);
                break;
            }
            // A negative number is an operand (`pen remote ADDRESS state -1`).
            let negative = word.strip_prefix('-').is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()));
            if !word.starts_with('-') || negative {
                args.operands.push(word);
                continue;
            }
            let (name, inline) = match word.split_once('=') {
                Some((name, value)) => (name, Some(value.to_owned())),
                None => (word.as_str(), None),
            };
            let name = if name == "-h" { "--help" } else { name };
            let spec = args.option(name)?;
            let value = match (spec.value, inline) {
                ("", None) => String::new(),
                ("", Some(_)) => bail!("{name} takes no value"),
                (_, Some(value)) => value,
                (placeholder, None) => words
                    .next_if(|value| !value.starts_with("--"))
                    .with_context(|| format!("{name} takes a value: {name} {placeholder}"))?,
            };
            args.options.push((spec.name, value));
        }
        let spec = COMMANDS.iter().find(|spec| spec.name == args.command);
        let operands = spec.map_or("", |spec| spec.operands);
        // Each word one operand, those in brackets optional, `...]` any number.
        let words: Vec<&str> = operands.split_whitespace().collect();
        let most = if operands.ends_with("...]") { usize::MAX } else { words.len() };
        let least = words.iter().filter(|word| !word.starts_with('[')).count();
        if let Some(extra) = args.operands.get(most) {
            bail!(
                "{} takes {}, not {extra}",
                invocation(args.command),
                if most == 0 { "no operands".into() } else { format!("{operands} and options") }
            );
        }
        ensure!(args.operands.len() >= least, "{} needs {operands}", invocation(args.command));
        let mut words = args.values("--script");
        for file in args.values("--script-file") {
            let text = std::fs::read_to_string(&file).with_context(|| format!("reading {file}"))?;
            words.extend(text.lines().filter(|line| !line.trim_start().starts_with('#')).map(str::to_owned));
        }
        args.script = script::parse(&words)?;
        Ok(args)
    }

    /// The option `name` of this command, or why there is none.
    fn option(&self, name: &str) -> Result<&'static OptionSpec> {
        let fits = |spec: &OptionSpec| spec.commands.is_empty() || spec.commands.contains(&self.command);
        if let Some(spec) = OPTIONS.iter().find(|spec| spec.name == name && fits(spec)) {
            return Ok(spec);
        }
        let elsewhere: Vec<String> = OPTIONS
            .iter()
            .filter(|spec| spec.name == name)
            .flat_map(|spec| spec.commands.iter().map(|command| invocation(command)))
            .collect();
        if elsewhere.is_empty() {
            let names = OPTIONS.iter().filter(|spec| fits(spec)).map(|spec| spec.name);
            bail!("{} has no option {name}{} (pen --help)", invocation(self.command), suggestion(name, names));
        }
        bail!("{name} is for {}, not {}", listed(&elsewhere), invocation(self.command))
    }

    fn known(name: &str) {
        debug_assert!(OPTIONS.iter().any(|spec| spec.name == name), "no option {name}");
    }

    fn flag(&self, name: &str) -> bool {
        Args::known(name);
        self.options.iter().any(|(option, _)| *option == name)
    }

    /// Every value given to `name`, in order.
    fn values(&self, name: &str) -> Vec<String> {
        Args::known(name);
        self.options.iter().filter(|(option, _)| *option == name).map(|(_, value)| value.clone()).collect()
    }

    /// The last value given to `name`.
    fn value(&self, name: &str) -> Option<String> {
        self.values(name).pop()
    }

    fn path(&self, name: &str) -> Option<PathBuf> {
        self.value(name).map(PathBuf::from)
    }

    fn number<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>> {
        self.value(name).map(|text| text.parse().map_err(|_| anyhow!("{name} takes a number, not {text}"))).transpose()
    }

    /// The dev tools are asked for: their console, a shot of them, or a
    /// script typing into them.
    fn overlay_asked(&self) -> bool {
        self.flag("--overlay")
            || self.flag("--overlay-shot")
            || self.script.iter().any(|step| matches!(step, script::Step::Overlay(_)))
    }
}

fn size(text: &str) -> Result<(u32, u32)> {
    let (width, height) = text.split_once('x').context("--size takes WIDTHxHEIGHT")?;
    let size = (width.parse()?, height.parse()?);
    if size.0 == 0 || size.1 == 0 {
        bail!("--size must not be empty");
    }
    Ok(size)
}

/// The mods `--mod` names, if any.
fn load_mods(args: &Args) -> Result<Option<mods::Mods>> {
    let directories: Vec<PathBuf> = args.values("--mod").into_iter().map(PathBuf::from).collect();
    if directories.is_empty() {
        return Ok(None);
    }
    mods::Mods::load(&directories).map(Some)
}

/// Where a game's program, assets and scene come from.
struct Game {
    /// The directory holding game.wasm (and perhaps game.cwasm).
    directory: PathBuf,
    assets: PathBuf,
    /// The scene the editor reads and saves; None plays the bundle's, saving
    /// the player's own in the data directory.
    scene: Option<PathBuf>,
}

impl Game {
    fn bundle(args: &Args, directory: PathBuf) -> Game {
        Game {
            assets: args.path("--assets").unwrap_or_else(|| directory.join("assets")),
            scene: args.path("--scene"),
            directory,
        }
    }

    /// A checkout's game, compiled into its `.pen/runtime`.
    fn checkout(project: &Project) -> Game {
        Game { directory: project.native(), assets: project.assets(), scene: Some(project.scene()) }
    }

    fn scene_to_start(&self) -> PathBuf {
        self.scene.clone().unwrap_or_else(|| self.directory.join("main.scene"))
    }
}

/// The packaged game to play: `--game DIR`, else `game/` beside the
/// executable.
fn bundle_directory(args: &Args) -> Result<PathBuf> {
    if let Some(directory) = args.path("--game").or_else(packaged_game) {
        return Ok(directory);
    }
    match std::env::current_dir().ok().and_then(|directory| Project::around(&directory)) {
        Some(root) => bail!(
            "there is no packaged game here: in the checkout at {}, `pen dev` or `pen run` plays its game \
             (pen --help)",
            root.display()
        ),
        None => bail!(
            "there is no game to play: pen plays the game/ folder beside its executable (a `pen build` \
             package) or --game DIR; in a checkout, `pen dev` or `pen run` (pen --help)"
        ),
    }
}

/// Says which of a script's `tune=` steps name no tunable of the checkout's
/// `src/tunables.json`: the game would take them and change nothing.
fn check_tunes(steps: &[script::Step], root: &Path) {
    let names: Vec<&str> = steps
        .iter()
        .filter_map(|step| match step {
            script::Step::Tune(name, _) => Some(name.as_str()),
            _ => None,
        })
        .collect();
    if names.is_empty() {
        return;
    }
    match tunables::load(root) {
        Ok(declared) => {
            for name in names.into_iter().filter(|name| declared.iter().all(|tunable| tunable.name != *name)) {
                eprintln!("script: tune={name} names no tunable in src/tunables.json, so it changes nothing");
            }
        }
        Err(problem) => eprintln!("script: {problem}"),
    }
}

/// `checkout` is the checkout played (`dev` and `run`), which has the dev
/// tools; `steam` starts Steam, as a package does.
fn play(args: &Args, game: Game, dev: bool, steam: bool, checkout: Option<&Path>, software: bool) -> Result<()> {
    let headless = args.flag("--headless");
    let frames: Option<usize> = args.number("--frames")?;
    let smoke = std::env::var("PEN_SMOKE_FRAMES").ok().map(|frames| frames.parse()).transpose()?;
    let screenshot = args.path("--screenshot");
    let overlay_shot = args.path("--overlay-shot");
    let record = args.path("--record");
    let script = args.script.clone();
    if let Some(root) = checkout {
        check_tunes(&script, root);
    }
    let settle = (headless || record.is_some() || args.flag("--settle")) && !args.flag("--no-settle");
    // A window always has the overlay; a headless run, when something asks for it.
    let overlay_asked = args.overlay_asked();
    if overlay_asked && checkout.is_none() {
        bail!("the dev tools are for a checkout, in `pen dev` and `pen run`");
    }
    #[cfg(not(feature = "devtools"))]
    if overlay_asked {
        bail!("this pen has no dev tools (it was built without the devtools feature)");
    }
    let shot = screenshot.is_some() || overlay_shot.is_some();
    // A frame lasts 1/fps, and a recording's sound fps times a second.
    let fps: Option<f64> = args.number("--fps")?;
    ensure!(fps.is_none_or(|fps| fps.is_finite() && fps > 0.0), "--fps takes a positive number");
    let frame_budget: Option<f32> = args.number("--frame-budget")?;
    ensure!(
        frame_budget.is_none_or(|budget| budget.is_finite() && budget > 0.0),
        "--frame-budget takes a positive number of milliseconds"
    );
    let net = args.value("--net").map(|spec| net::Spec::parse(&spec)).transpose()?;
    let spectating = args.value("--spectate").is_some();
    if let Some(spec) = &net {
        net::check_players(identity().players, spec.addresses.len())?;
        ensure!(args.path("--replay").is_none(), "a net session plays its players' input, not a recording's");
    }
    ensure!(
        args.value("--spectate").is_none() || (net.is_none() && args.path("--replay").is_none()),
        "--spectate plays a stream's input, not a net session's or a recording's"
    );
    // A net session's frames are a steady 1/60 s each, the host's.
    let fps = fps.or(net.as_ref().map(|_| 60.0));
    let options = Options {
        // A viewer plays until the stream ends.
        frames: frames.or(smoke).unwrap_or(if shot && script.is_empty() && !spectating { 120 } else { 0 }),
        screenshot,
        record,
        fps,
        mute: args.flag("--mute"),
        audio_buffer: match args.number("--audio-buffer")? {
            Some(samples) => Some(samples),
            None => std::env::var("PEN_AUDIO_BUFFER").ok().map(|samples| samples.parse()).transpose()?,
        },
        dev,
        script,
        headless,
        settle,
    };
    let playback = Playback {
        replay: args.path("--replay"),
        record_input: args.path("--record-input"),
        history: args.flag("--history"),
        overlap: args.flag("--overlap"),
        trace: args.path("--trace"),
        net,
        stream: args.value("--broadcast"),
        spectate: args.value("--spectate"),
    };
    if headless
        && options.frames == 0
        && options.script.is_empty()
        && playback.replay.is_none()
        && playback.spectate.is_none()
    {
        bail!("--headless needs --frames N, a --script, a --replay or a --spectate to know when to stop");
    }
    // A replay draws at the size it was recorded at, unless told otherwise.
    let recorded_size = match &playback.replay {
        Some(path) => recording::viewport(path)?,
        None => None,
    };
    let size = args.value("--size").map(|text| size(&text)).transpose()?.or(recorded_size);
    let paths = Paths::new(
        game.directory.clone(),
        Some(game.assets.clone()),
        game.scene.clone(),
        args.path("--shots"),
        args.path("--data"),
    );
    // What the scene draws first is read while the program loads and the GPU opens.
    scene_assets::warm_up(&paths.assets, paths.scene_to_read());
    // Development recompiles after every edit, reusing unchanged functions.
    let engine = if dev { blot_host::dev_engine()? } else { blot_host::engine()? };
    let module = load_module(&engine, &game.directory, !dev, checkout.is_some())?;
    let catalog = read_catalog(&paths.assets)?;
    let mods = load_mods(args)?;
    // A packaged game has no dev tools program, but may answer remote ones.
    let listen = args.value("--devtools-listen");
    #[cfg(not(feature = "devtools"))]
    if listen.is_some() {
        bail!("this pen has no dev tools (it was built without the devtools feature)");
    }
    #[cfg(feature = "devtools")]
    let devtools = |keep: bool| {
        let root = checkout.map(Path::to_owned).or_else(|| listen.is_some().then(|| game.directory.clone()))?;
        Some(devtools::Settings {
            project: root,
            build: game.directory.clone(),
            open: devtools::requested(args.flag("--overlay")),
            keep: keep && checkout.is_some(),
            listen: listen.clone(),
        })
    };

    if headless {
        let mut gpu = Gpu::headless(size.unwrap_or((1920, 1280)), &paths.assets, software)?;
        gpu.renderer.frame_budget(frame_budget);
        #[cfg(feature = "devtools")]
        let (assets, devtools_catalog) = (paths.assets.clone(), catalog.clone());
        let mut session = Session::start(options, paths, engine, module, catalog, gpu, None)?;
        session.playback(playback, None)?;
        if let Some(mods) = mods {
            session.set_mods(mods);
        }
        #[cfg(feature = "devtools")]
        if let Some(settings) = devtools(false).filter(|_| overlay_asked || listen.is_some()) {
            let mut devtools = devtools::Devtools::new(None, &session.gpu, devtools_catalog, &assets, settings)?;
            // Headless, the program's build is waited for: what it shows is the point.
            if overlay_asked {
                devtools.start_now()?;
            }
            session.set_overlay(Box::new(devtools));
        }
        if let Some(path) = overlay_shot {
            session.overlay_shot_at_end(path);
        }
        let mut accessibility = accessibility::Accessibility::headless()?;
        let started = Instant::now();
        let mut frames = 0usize;
        let result = loop {
            frames += 1;
            let stepped = Instant::now();
            let step = session.step(None);
            crash::frame(stepped.elapsed());
            if let (Ok(_), Some(accessibility)) = (&step, &mut accessibility) {
                accessibility.frame(&mut session, None);
            }
            match step {
                Ok(true) => {}
                Ok(false) => break Ok(()),
                Err(error) => break Err(error),
            }
        };
        let finished = session.finish().map(drop);
        let seconds = started.elapsed().as_secs_f64();
        eprintln!("headless: {frames} frames in {seconds:.1} s ({:.0} fps)", frames as f64 / seconds);
        return result.and(finished);
    }

    let event_loop = EventLoop::new()?;
    let mut app = App {
        pending: Some(Pending {
            options,
            playback,
            paths,
            engine,
            module,
            catalog,
            present_mode: present_mode()?,
            size,
            software,
            frame_budget,
            #[cfg(feature = "devtools")]
            devtools: devtools(true),
            overlay_shot,
            mods,
        }),
        window: None,
        session: None,
        accessibility: None,
        failure: None,
        #[cfg(feature = "steam")]
        steam: (steam || steam::requested()).then(steam::Steam::init).flatten(),
    };
    #[cfg(not(feature = "steam"))]
    let _ = steam;
    event_loop.run_app(&mut app)?;
    match app.failure.take() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// `pen new DIR`.
#[cfg(feature = "compiler")]
fn new_game(args: &Args) -> Result<()> {
    let root = new::create(Path::new(&args.operands[0]), args.path("--blot"))?;
    println!(
        "made {} in {}: `pen dev` there plays it as you edit it, `pen test-blot` runs its tests",
        project::Project::find(Some(root.clone()))?.identity.name,
        root.display()
    );
    Ok(())
}

#[cfg(not(feature = "compiler"))]
fn new_game(_: &Args) -> Result<()> {
    bail!("this player runtime makes no games; use a full pen build")
}

/// The commands that work over the network, which need no checkout: none
/// for another command.
fn network(args: &Args) -> Option<Result<()>> {
    #[cfg(feature = "compiler")]
    let token = || args.value("--token").or_else(|| std::env::var("PEN_PUSH_TOKEN").ok());
    match args.command {
        #[cfg(feature = "compiler")]
        "serve" => Some((|| {
            let token = token().unwrap_or_else(|| {
                let made = push::new_token();
                eprintln!("serve: the token is {made} (pen push --token {made})");
                made
            });
            let serve = push::Serve {
                listen: args.value("--listen").unwrap_or_else(|| format!("127.0.0.1:{}", push::PORT)),
                token,
                directory: args.path("--dir").unwrap_or_else(|| data_dir().join("pushed")),
                allow_run: args.flag("--allow-run"),
            };
            let listener =
                std::net::TcpListener::bind(&serve.listen).with_context(|| format!("listening on {}", serve.listen))?;
            eprintln!("serve: packages pushed to {} go in {}", listener.local_addr()?, serve.directory.display());
            push::Server::default().serve(&serve, &listener, || false)
        })()),
        #[cfg(feature = "compiler")]
        "push" => Some((|| {
            let token = token().context("pen push needs the server's token: --token TEXT or PEN_PUSH_TOKEN")?;
            let package = args.path("--package").unwrap_or_else(|| PathBuf::from(".pen/package"));
            let name = match args.value("--name") {
                Some(name) => name,
                None => project::bundle_identity(&package.join("game"))
                    .map(|identity| identity.name)
                    .with_context(|| format!("{} is not a package (pen build makes one)", package.display()))?,
            };
            let started = Instant::now();
            let done = push::push(&push::Push {
                address: &args.operands[0],
                token: &token,
                package: &package,
                name: &name,
                run: args.value("--run").as_deref(),
            })?;
            eprintln!(
                "push: {} of {} files ({:.1} MB) to {} in {:.1} s{}",
                done.sent,
                done.files,
                done.bytes as f64 / 1e6,
                done.at,
                started.elapsed().as_secs_f64(),
                done.started.map(|run| format!("; started {run}")).unwrap_or_default()
            );
            Ok(())
        })()),
        #[cfg(not(feature = "compiler"))]
        "serve" | "push" => Some(Err(anyhow!("this player runtime neither serves nor pushes packages"))),
        #[cfg(feature = "devtools")]
        "remote" => Some(remote::command(
            &args.operands[0],
            &args.operands[1..],
            args.value("--text").as_deref(),
            args.flag("--follow"),
        )),
        #[cfg(not(feature = "devtools"))]
        "remote" => Some(Err(anyhow!("this pen has no dev tools to ask a game remotely"))),
        _ => None,
    }
}

fn main() -> ExitCode {
    project::keep_mark();
    startup::begin();
    logger::install();
    let result = run(std::env::args().skip(1));
    reports::finish();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            crash::stopped(&error);
            ExitCode::FAILURE
        }
    }
}

/// The tag and lanes of the game's own event `name`, as the `pen.json` of
/// the checkout at `root` declares it, for `game=` steps.
fn game_event(root: Option<&Path>, name: &str, values: &[f32]) -> Result<(u32, Vec<f32>)> {
    #[cfg(feature = "compiler")]
    if let Some(root) = root {
        let declarations = pen_codegen::extensions::Declarations::of_checkout(root)?;
        let event = declarations
            .event(name)
            .with_context(|| format!("{} declares no event {name}", root.join("pen.json").display()))?;
        return Ok((event.tag, event.lanes(values)?));
    }
    let _ = (root, values);
    bail!("only a checkout's pen.json names the game's events: give {name}'s tag")
}

/// One of the game's own sections, as `pen packets` writes it: by its
/// declaration in the `pen.json` of the checkout at `root` (its name and
/// records) when it has one, else by its tag and lanes.
fn section_describer(root: Option<&Path>) -> Result<impl Fn(&pen_protocol::Extension) -> serde_json::Value> {
    #[cfg(feature = "compiler")]
    let declarations = match root {
        Some(root) => pen_codegen::extensions::Declarations::of_checkout(root)?,
        None => pen_codegen::extensions::Declarations::default(),
    };
    #[cfg(not(feature = "compiler"))]
    let _ = root;
    Ok(move |extension: &pen_protocol::Extension| {
        #[cfg(feature = "compiler")]
        if let Ok(described) = declarations.describe(extension.tag, &extension.values) {
            return described;
        }
        serde_json::json!({ "tag": extension.tag, "values": extension.values })
    })
}

/// `pen texture` and `pen font`, which need no checkout.
#[cfg(feature = "compiler")]
fn asset_tool(args: &Args) -> Result<()> {
    let source = PathBuf::from(&args.operands[0]);
    if args.command == "texture" {
        return textures::texture(&source, args.path("--out"), args.flag("--linear"), !args.flag("--flat"));
    }
    let options = textures::FontOptions {
        em: args.number("--em")?.unwrap_or(48.0),
        range: args.number("--range")?.unwrap_or(4.0),
        width: args.number("--width")?.unwrap_or(512),
        weight: args.number("--weight")?,
        chars: args.value("--chars"),
    };
    textures::font(&source, args.path("--out"), options)
}

#[cfg(not(feature = "compiler"))]
fn asset_tool(args: &Args) -> Result<()> {
    bail!("this player runtime has no {} tool", args.command)
}

fn run(words: impl IntoIterator<Item = String>) -> Result<()> {
    let mut args = Args::parse(words)?;
    if args.flag("--help") {
        println!("{}", usage());
        return Ok(());
    }
    if args.flag("--version") {
        #[cfg(feature = "compiler")]
        let plugins = format!(", plugins: {}@{}", pen_plugins::api::PACKAGE, pen_plugins::api::VERSION);
        #[cfg(not(feature = "compiler"))]
        let plugins = "";
        println!("pen {} (game runtime: {}{plugins})", env!("CARGO_PKG_VERSION"), project::RUNTIME);
        return Ok(());
    }
    if args.command == "clean" {
        return clean();
    }
    if args.command == "gltf" {
        return import_gltf(&args);
    }
    if args.command == "report" {
        return bugreport::open(Path::new(&args.operands[0]), args.path("--out").as_deref());
    }
    if args.command == "new" {
        return new_game(&args);
    }
    if let "texture" | "font" = args.command {
        return asset_tool(&args);
    }
    if let Some(network) = network(&args) {
        return network;
    }
    if args.command == "patch" {
        let store = args.path("--store").or_else(|| {
            let directory = std::env::current_dir().ok()?;
            Some(Project::around(&directory)?.join(".pen/store"))
        });
        let out = args.path("--out").context("patch needs --out FILE")?;
        let made = patch::make(Path::new(&args.operands[0]), Path::new(&args.operands[1]), store.as_deref(), &out)?;
        eprintln!("patch {}: {made}", out.display());
        return Ok(());
    }
    if args.command == "apply" {
        let package = match args.operands.get(1) {
            Some(directory) => PathBuf::from(directory),
            None => packaged_game()
                .and_then(|game| game.parent().map(Path::to_owned))
                .context("there is no package beside this executable: apply PATCH DIR")?,
        };
        let version = patch::apply(Path::new(&args.operands[0]), &package)?;
        eprintln!("{} is now version {version}", package.display());
        return Ok(());
    }
    // Sound files, anywhere: no checkout.
    if args.command == "loudness" {
        let files: Vec<PathBuf> = args.operands.iter().map(PathBuf::from).collect();
        let options = loudness::Options {
            normalize: args.flag("--normalize"),
            target: args.number("--lufs")?,
            peak: args.number("--ceiling")?,
            targets: args.path("--targets"),
        };
        return loudness::run(&files, &options);
    }
    // The checkout, for its commands but those given a game bundle instead.
    let bundled = args.command.is_empty()
        || (["build", "test", "bench", "packets", "diverge", "fuzz"].contains(&args.command) && args.flag("--game"));
    let (project, bundle) = if bundled {
        let directory = bundle_directory(&args)?;
        let _ = IDENTITY.set(project::bundle_identity(&directory)?);
        (None, Some(directory))
    } else {
        let project = if args.command == "tool" || (args.command == "test-blot" && !args.flag("--session")) {
            Project::find_tests(args.path("--project"))?
        } else {
            Project::find(args.path("--project"))?
        };
        let _ = IDENTITY.set(project.identity.clone());
        (Some(project), None)
    };
    let root = project.as_ref().map(|project| project.root.clone());
    script::name_game_events(&mut args.script, &|name, values| game_event(root.as_deref(), name, values))?;
    crash::install(args.path("--data"));
    // Only a player's windowed game has someone to show an error to.
    if args.flag("--headless") || !args.command.is_empty() {
        crash::unattended();
    }
    if let (true, Some(directory)) = (args.command.is_empty(), &bundle) {
        reports::install(directory, args.path("--data").unwrap_or_else(data_dir));
    }
    let software = args.flag("--software") || env_flag("PEN_SOFTWARE");
    // Every command but playing a package has its checkout, or its bundle.
    let checkout = || project.as_ref().context("no checkout");
    let game = |project: Option<&Project>| match (project, &bundle) {
        (Some(project), _) => Game::checkout(project),
        (None, bundle) => Game::bundle(&args, bundle.clone().unwrap_or_default()),
    };
    match args.command {
        "" => play(&args, game(None), false, packaged_game().is_some(), None, software),
        "dev" => {
            let project = checkout()?;
            // The dev tools' program builds beside the game's.
            let _devtools = project.build_devtools(&project.native())?;
            let _compiler = project.watch(&project.native())?;
            let _pipeline = (!args.flag("--no-pipeline")).then(|| pipeline::Pipeline::start(&project.root)).flatten();
            play(&args, Game::checkout(project), true, false, Some(&project.root), software)
        }
        "web" => {
            let project = project.context("no checkout")?;
            web::serve(project, args.number("--port")?)
        }
        "export" => project::export_bundle(checkout()?, &args.path("--out").context("export needs --out DIR")?),
        "check" => {
            let project = checkout()?;
            project.compile(&project.native())?;
            project::check_saved_types(&project.assets())?;
            project.compile_devtools(&project.native())?;
            let wasm = std::fs::read(project.native().join("game.wasm"))?;
            asset_lint::check(&project.root, &project.assets(), &wasm)
        }
        "assets" => {
            let project = checkout()?;
            let cache = match args.path("--cache") {
                Some(directory) => directory,
                None => asset_graph::default_cache().context("this system has no cache directory: pass --cache DIR")?,
            };
            let options = asset_graph::Options { cache, check: args.flag("--check"), only: args.values("--only") };
            asset_graph::summarize(&asset_graph::run(&project.root, &options)?, options.check)
        }
        #[cfg(feature = "compiler")]
        "codegen" => pen_codegen::generate(&checkout()?.root, args.path("--rust").as_deref(), args.flag("--check")),
        #[cfg(not(feature = "compiler"))]
        "codegen" => bail!("this player runtime has no protocol generator"),
        "run" => {
            let project = checkout()?;
            let headless = args.flag("--headless");
            let mut devtools =
                if !headless || args.overlay_asked() { project.build_devtools(&project.native())? } else { None };
            project.compile(&project.native())?;
            // Headless, the dev tools are wanted from the first frame.
            if headless
                && let Some(build) = &mut devtools
                && !build.wait()?
            {
                eprintln!("devtools: its program did not compile");
            }
            play(&args, Game::checkout(project), false, false, Some(&project.root), software)
        }
        "build" => {
            let options = project::PackageOptions {
                optimize: !args.flag("--no-optimize"),
                runtime: args.path("--runtime"),
                target: args.value("--target"),
                cpu: args.value("--cpu"),
                bundle_id: args.value("--bundle-id"),
                store: args.path("--store"),
            };
            // A bundle compiled already packages as it is, for another platform's runtime.
            if let Some(bundle) = &bundle {
                ensure!(!args.flag("--web"), "pen build --web exports a checkout, not --game");
                ensure!(!args.flag("--zip"), "--zip goes with --web");
                let out = args.path("--out").context("pen build --game needs --out DIR")?;
                return project::package(project::GameSource::Bundle(bundle), &out, options);
            }
            let project = checkout()?;
            let stores = publish::Request {
                steam: args.flag("--steam"),
                itch: args.flag("--itch"),
                upload: args.flag("--upload"),
                label: args.value("--label"),
            }
            .plan(project)?;
            let web = args.flag("--web");
            let out = if web {
                let out = args.path("--out").unwrap_or_else(|| project.root.join(".pen/web-export"));
                project::export_web(project, &out, args.path("--zip").as_deref(), options.optimize)?;
                out
            } else {
                ensure!(!args.flag("--zip"), "--zip goes with --web");
                let out = args.path("--out").unwrap_or_else(|| project.root.join(".pen/package"));
                project::package(project::GameSource::Checkout(project), &out, options)?;
                out
            };
            match stores {
                Some(stores) => stores.publish(project, &out, web),
                None => Ok(()),
            }
        }
        "test" => {
            let root = match &project {
                Some(project) => project.root.clone(),
                None => std::env::current_dir()?,
            };
            let file =
                args.operands.first().map(PathBuf::from).unwrap_or_else(|| root.join("tests/visual/scenarios.json"));
            let golden =
                args.path("--golden").unwrap_or_else(|| file.parent().unwrap_or(Path::new(".")).join("golden"));
            let out = args.path("--out").unwrap_or_else(|| root.join(".pen/visual"));
            let review = args.flag("--review").then(|| args.number("--port")).transpose()?;
            // The last run's review again: nothing runs, nothing draws.
            if args.flag("--last") {
                let port = review.context("--last is for --review")?;
                return review::serve(&out, &golden, review::last(&out)?, port);
            }
            if let Some(project) = &project {
                project.compile(&project.native())?;
            }
            let game = game(project.as_ref());
            let engine = blot_host::engine()?;
            let module = load_module(&engine, &game.directory, true, false)?;
            let catalog = read_catalog(&game.assets)?;
            let test = visual::TestOptions {
                checkout: project.as_ref().map(|project| project.root.clone()),
                golden,
                out,
                file,
                update: args.flag("--update"),
                only: args.values("--only"),
                software,
                review,
            };
            let scene = game.scene_to_start();
            let visual_game = visual::Game { directory: &game.directory, assets: &game.assets, scene: &scene };
            visual::run(test, visual_game, engine, module, catalog)
        }
        command @ ("bench" | "packets") => {
            if let Some(project) = &project {
                project.compile(&project.native())?;
            }
            let game = game(project.as_ref());
            let engine = blot_host::engine()?;
            let bench = command == "bench";
            // A bench times loading the module, so not from the cache.
            let loading = Instant::now();
            let module = load_module(&engine, &game.directory, !bench, false)?;
            let load_ms = loading.elapsed().as_secs_f64() * 1000.0;
            let frames: Option<usize> = args.number("--frames")?;
            let out = (!bench).then(|| PathBuf::from(&args.operands[0]));
            let frames = match (command, args.script.is_empty()) {
                ("bench", _) => Some(frames.unwrap_or(600)),
                (_, true) => Some(frames.unwrap_or(120)),
                _ => frames,
            };
            let describe = section_describer(root.as_deref())?;
            let run = |out: Option<PathBuf>| {
                packets::run(packets::Run {
                    module: &module,
                    engine: &engine,
                    assets: game.assets.clone(),
                    scene: game.scene_to_start(),
                    viewport: Viewport { width: 1280, height: 720 },
                    out,
                    frames,
                    steps: args.script.clone(),
                    mods: load_mods(&args)?,
                    snapshots: bench,
                    describe: &describe,
                })
            };
            if !bench {
                return run(out).map(drop);
            }
            let runs: usize = args.number("--runs")?.unwrap_or(1);
            ensure!(runs > 0, "--runs takes 1 or more");
            let measured = (0..runs).map(|_| run(None)).collect::<Result<Vec<_>>>()?;
            let wasm = std::fs::read(game.directory.join("game.wasm"))?;
            let results = bench::results(&wasm, load_ms, &measured)?;
            if args.flag("--json") {
                println!("{}", serde_json::to_string_pretty(&results)?);
            }
            let bases: Vec<PathBuf> = args.values("--compare").into_iter().map(PathBuf::from).collect();
            ensure!(!bases.is_empty() || args.value("--threshold").is_none(), "--threshold is for --compare");
            if bases.is_empty() { Ok(()) } else { bench::check(&bases, &results, args.number("--threshold")?) }
        }
        "diverge" => {
            if let Some(project) = &project {
                project.compile(&project.native())?;
            }
            let game = game(project.as_ref());
            let against = args.path("--against").context("diverge needs --against DIR, the build to compare with")?;
            let reference = diverge::Bundle { assets: against.join("assets"), directory: against };
            let this = diverge::Bundle { directory: game.directory, assets: game.assets };
            diverge::run(Path::new(&args.operands[0]), &this, &reference)
        }
        "fuzz" => {
            if let Some(project) = &project {
                project.compile(&project.native())?;
            }
            let game = game(project.as_ref());
            let fuzzed = fuzz::Game::load(blot_host::engine()?, &game.directory, &game.assets, &game.scene_to_start())?;
            let frames: usize = args.number("--frames")?.unwrap_or(600);
            // A script given (one a run was cut down to): the same checks on it.
            if !args.script.is_empty() {
                return match fuzz::check(&fuzzed, args.script.clone(), frames)? {
                    Some(broken) => {
                        bail!(
                            "the script breaks \"{}\" at frame {}: {}",
                            broken.invariant,
                            broken.frame,
                            broken.message
                        )
                    }
                    None => {
                        eprintln!("fuzz: the script broke nothing in {frames} frames");
                        Ok(())
                    }
                };
            }
            let tunables = match &project {
                Some(project) => tunables::load(&project.root).unwrap_or_default(),
                None => Vec::new(),
            };
            let root = project.as_ref().map(|project| project.root.clone());
            let options = fuzz::Options {
                seed: args.number("--seed")?.unwrap_or(1),
                runs: args.number("--runs")?.unwrap_or(10),
                frames,
                out: args.path("--out").unwrap_or_else(|| root.unwrap_or_default().join(".pen/fuzz")),
                vocabulary: fuzz::Vocabulary {
                    tunables: tunables.into_iter().map(|tunable| (tunable.name, tunable.min, tunable.max)).collect(),
                },
            };
            fuzz::run(&fuzzed, &options, None).map(drop)
        }
        #[cfg(feature = "compiler")]
        "tool" => {
            let project = checkout()?;
            let plugins =
                pen_plugins::Plugins::load(&project.root, Arc::new(|level, text| eprintln!("{level:?}: {text}")))?
                    .context("pen tool needs a pen.json")?;
            let name = &args.operands[0];
            let tool = plugins.config().tools.get(name).with_context(|| format!("no tool {name} in pen.json"))?;
            let manifest = project.root.join(&tool.manifest).canonicalize().context("tool manifest")?;
            ensure!(manifest.starts_with(&project.root), "tool manifest must be inside the checkout");
            let status = std::process::Command::new("cargo")
                .args(["run", "--quiet", "--locked", "--manifest-path"])
                .arg(&manifest)
                .arg("--bin")
                .arg(&tool.bin)
                .arg("--")
                .args(&args.operands[1..])
                .current_dir(&project.root)
                .env("PEN_EXECUTABLE", std::env::current_exe()?)
                .env("PEN_PROJECT", &project.root)
                .status()
                .context("starting checkout tool with Cargo")?;
            ensure!(status.success(), "tool {name} failed: {status}");
            Ok(())
        }
        #[cfg(not(feature = "compiler"))]
        "tool" => bail!("this pen has no checkout tools; use a full build"),
        #[cfg(feature = "compiler")]
        "test-blot" => {
            let project = checkout()?;
            let files = args.operands.iter().map(PathBuf::from).collect();
            let deadline = args.number::<u64>("--deadline")?.map(std::time::Duration::from_secs);
            let session = if args.flag("--session") {
                ensure!(!args.operands.is_empty(), "--session requires explicit Blot scenario files");
                Some(blot_test::SessionOptions {
                    out: args.path("--out"),
                    golden: args.path("--golden"),
                    update: args.flag("--update"),
                    software: args.flag("--software") || std::env::var_os("PEN_SOFTWARE").is_some_and(|v| v == "1"),
                })
            } else {
                ensure!(
                    !args.flag("--update")
                        && !args.flag("--software")
                        && args.path("--out").is_none()
                        && args.path("--golden").is_none(),
                    "--out, --golden, --update and --software need --session for test-blot"
                );
                None
            };
            let options = blot_test::Options {
                files,
                filter: args.value("--filter"),
                jobs: args.number("--jobs")?,
                deadline,
                session,
            };
            blot_test::run(&project.root, options)
        }
        #[cfg(not(feature = "compiler"))]
        "test-blot" => bail!("this pen has no plugins, which compile a checkout's Blot tests"),
        other => unreachable!("Args::parse takes only known commands, and run the rest earlier, not {other}"),
    }
}

/// `pen gltf FILE`: the file's assets into `--out` (gltf.rs).
#[cfg(feature = "compiler")]
fn import_gltf(args: &Args) -> Result<()> {
    let options = pen_gltf::Options {
        name: args.value("--name"),
        shader: args.value("--shader").unwrap_or_else(|| "scene.wgsl".to_owned()),
        vat: args.number("--vat")?.map(|fps| pen_gltf::VatOptions {
            fps,
            shader: args.value("--vat-shader").unwrap_or_else(|| "vat.wgsl".to_owned()),
        }),
    };
    let out = args.path("--out").unwrap_or_else(|| PathBuf::from("src/assets"));
    gltf::import(Path::new(&args.operands[0]), &out, &options)
}

#[cfg(not(feature = "compiler"))]
fn import_gltf(_: &Args) -> Result<()> {
    bail!("this player runtime has no asset importer")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> Result<Args> {
        Args::parse(line.split_whitespace().map(str::to_owned))
    }

    fn error(line: &str) -> String {
        format!("{:#}", parse(line).unwrap_err())
    }

    #[test]
    fn checkout_tools_forward_arguments_after_the_separator_verbatim() {
        let args = parse("tool ink --project ../ink -- generate config.json --write --project ../gdev").unwrap();
        assert_eq!(args.command, "tool");
        assert_eq!(args.value("--project").as_deref(), Some("../ink"));
        assert_eq!(args.operands, ["ink", "generate", "config.json", "--write", "--project", "../gdev"]);
        assert!(parse("tool").is_err());
        assert!(parse("tool ink --unknown").is_err());
    }

    #[test]
    fn arguments_parse_into_a_command_operands_and_options() {
        let args = parse("run --headless --frames 300 --size=640x480 --script wait=2 --script quit").unwrap();
        assert_eq!(args.command, "run");
        assert!(args.flag("--headless") && !args.flag("--mute"));
        assert_eq!(args.number::<usize>("--frames").unwrap(), Some(300));
        assert_eq!(args.value("--size").as_deref(), Some("640x480"));
        assert_eq!(args.values("--script"), ["wait=2", "quit"]);
        assert_eq!(args.script.len(), 2);
        // Without a command, pen plays a package.
        let args = parse("--game dist/gdev -h").unwrap();
        assert_eq!((args.command, args.path("--game")), ("", Some(PathBuf::from("dist/gdev"))));
        assert!(args.flag("--help"));
        assert_eq!(parse("test scenarios.json --only rock --only menu").unwrap().values("--only"), ["rock", "menu"]);
        assert_eq!(parse("test-blot a.blot b.blot --jobs 2").unwrap().operands, ["a.blot", "b.blot"]);
        assert_eq!(parse("packets out --frames 5").unwrap().operands, ["out"]);
        let args = parse("gltf models/a.glb --vat 30 --out assets").unwrap();
        assert_eq!(
            (args.operands.as_slice(), args.number::<f32>("--vat").unwrap()),
            (&["models/a.glb".to_owned()][..], Some(30.0))
        );
        assert!(parse("gltf").is_err() && parse("gltf a.glb --frames 3").is_err());
        assert_eq!(parse("patch v1 v2 --out up.penpatch").unwrap().operands, ["v1", "v2"]);
        assert_eq!(parse("apply up.penpatch").unwrap().operands, ["up.penpatch"]);
        assert!(parse("--version").unwrap().flag("--version"));
        assert!(parse("clean").is_ok());
        let args = parse("push deck.local:7811 --token t --run gdev").unwrap();
        assert_eq!(
            (args.operands.as_slice(), args.value("--run").as_deref()),
            (&["deck.local:7811".to_owned()][..], Some("gdev"))
        );
        assert_eq!(parse("remote 127.0.0.1 watch 100 3 3 5").unwrap().operands.len(), 6);
        assert_eq!(parse("remote 127.0.0.1 state -1").unwrap().operands[2], "-1");
        assert!(parse("serve --listen 0.0.0.0:7811 --allow-run").unwrap().flag("--allow-run"));
        assert!(parse("run --devtools-listen 127.0.0.1:7812").is_ok());
    }

    #[test]
    fn mistakes_are_refused_with_what_was_meant() {
        assert_eq!(
            error("tset"),
            "unknown command tset (did you mean test?); the commands are dev, web, run, build, check, assets, codegen, export, test, bench, packets, test-blot, gltf, diverge, fuzz, report, new, texture, font, patch, apply, loudness, clean, serve, push and remote (pen --help)"
        );
        assert_eq!(error("run --frame 300"), "pen run has no option --frame (did you mean --frames?) (pen --help)");
        assert_eq!(error("run --port 8000"), "--port is for pen web and pen test, not pen run");
        assert_eq!(
            error("build --frames 3"),
            "--frames is for pen, pen dev, pen run, pen bench, pen packets and pen fuzz, not pen build"
        );
        assert_eq!(error("--steam"), "--steam is for pen build, not pen");
        assert_eq!(error("--stem"), "pen has no option --stem (pen --help)");
        assert_eq!(error("dev --overlay=1"), "--overlay takes no value");
        assert_eq!(error("run --frames"), "--frames takes a value: --frames N");
        assert_eq!(error("build --out --web"), "--out takes a value: --out DIR");
        assert_eq!(error("run extra"), "pen run takes no operands, not extra");
        assert_eq!(error("test a.json b.json"), "pen test takes [FILE] and options, not b.json");
        assert_eq!(error("packets"), "pen packets needs DIR");
        assert_eq!(error("patch v1"), "pen patch needs OLD NEW");
        assert_eq!(error("apply a b c"), "pen apply takes PATCH [DIR] and options, not c");
        assert_eq!(error("-x"), "pen has no option -x (pen --help)");
        assert_eq!(error("run --script bogus=1"), "unknown script step bogus=1");
        let args = parse("run --frames many").unwrap();
        assert_eq!(format!("{:#}", args.number::<usize>("--frames").unwrap_err()), "--frames takes a number, not many");
    }

    #[test]
    fn the_dev_tools_are_asked_for_once() {
        assert!(!parse("run --headless --frames 1").unwrap().overlay_asked());
        assert!(parse("run --headless --overlay --frames 1").unwrap().overlay_asked());
        assert!(parse("run --headless --overlay-shot a.png").unwrap().overlay_asked());
        assert!(parse("run --headless --script overlay=/help").unwrap().overlay_asked());
    }

    #[test]
    fn the_usage_tells_of_every_option_and_command() {
        let text = usage();
        for spec in OPTIONS.iter().filter(|spec| spec.place != Place::Prose) {
            assert!(text.contains(&format!("{} {}", spec.name, spec.value).trim_end().to_owned()), "{}", spec.name);
        }
        for spec in COMMANDS {
            assert!(text.contains(&format!("  {}", spec.name)), "{}", spec.name);
        }
        assert!(text.lines().all(|line| line.chars().count() <= 100), "{text}");
        // An option's name means one kind of thing, whichever command it is for.
        for spec in OPTIONS {
            for other in OPTIONS.iter().filter(|other| other.name == spec.name) {
                assert_eq!(spec.value.is_empty(), other.value.is_empty(), "{}", spec.name);
            }
        }
        // Every variable the crate reads is in it.
        for name in [
            "PEN_PRESENT_MODE",
            "PEN_SOFTWARE",
            "PEN_FRAME_TIMES",
            "PEN_RECORD_PCM",
            "PEN_PLUGINS",
            "PEN_ASSET_CACHE",
            "PEN_STARTUP",
            "PEN_WEB_HOST",
            "PEN_STEAM_APP_ID",
            "PEN_CRASH_TEST",
            "BLOT_INLINING",
            "STEAM_BUILD_USER",
            "PEN_STEAMCMD",
            "PEN_BUTLER",
            "BUTLER_API_KEY",
            "PEN_GRAPHICS_TIER",
            "SOURCE_DATE_EPOCH",
            "PEN_REPORTS",
        ] {
            assert!(text.contains(name), "{name}");
        }
    }

    #[test]
    fn names_are_suggested_when_near() {
        assert_eq!(distance("tset", "test"), 2);
        assert_eq!(distance("", "abc"), 3);
        assert_eq!(suggestion("--frame", ["--fps", "--frames"].into_iter()), " (did you mean --frames?)");
        assert_eq!(suggestion("--bogus", ["--fps", "--frames"].into_iter()), "");
    }
}
