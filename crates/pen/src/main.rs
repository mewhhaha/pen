//! Native front end: windows, rendering, audio, headless verification and packaging.
// The player runtime opens no console window on Windows (an error it stops
// with goes to a crash report and a message box); the development one keeps
// its console.
#![cfg_attr(all(windows, not(feature = "compiler")), windows_subsystem = "windows")]
mod accessibility;
mod audio;
#[cfg(feature = "compiler")]
mod blot_test;
mod crash;
mod dev;
mod devlog;
#[cfg(feature = "devtools")]
mod devtools;
#[cfg(test)]
mod fixtures;
mod game;
mod input;
mod logger;
mod opus;
mod packets;
mod pipeline;
#[cfg(feature = "compiler")]
mod plugin_build;
mod program;
mod project;
mod record;
mod recording;
mod script;
mod services;
mod session;
mod snapshot;
mod staging;
#[cfg(feature = "steam")]
mod steam;
mod timeline;
mod tunables;
mod visual;
mod web;

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
/// compiled pipelines, and tests' scratch files.
const CACHES: [&str; 6] = ["wasm", "components", "programs", "plugins", "pipelines", "tmp"];

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
fn load_module(engine: &wasmtime::Engine, game: &Path, cache: bool) -> Result<GuestModule> {
    let path = game.join("game.wasm");
    let wasm = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let cache = if cache { cache_dir().map(|directory| directory.join("wasm")) } else { None };
    let started = Instant::now();
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
    /// The dev tools, in `dev` and `run`.
    #[cfg(feature = "devtools")]
    devtools: Option<devtools::Settings>,
    /// Write the last frame with the dev tools (`--overlay-shot`).
    overlay_shot: Option<PathBuf>,
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
        let gpu = Gpu::windowed(
            Box::new(event_loop.owned_display_handle()),
            window.clone(),
            pending.present_mode,
            &pending.paths.assets,
            pinned,
            pending.software,
        )?;
        #[cfg(feature = "devtools")]
        let (assets, catalog) = (pending.paths.assets.clone(), pending.catalog.clone());
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
        // The game's achievements, stats and rich presence go through its
        // services; the session pumps Steam's callbacks each frame.
        #[cfg(feature = "steam")]
        session.services.register(Box::new(services::steam::SteamService::new(self.steam.take())));
        #[cfg(feature = "devtools")]
        if let Some(settings) = pending.devtools {
            let devtools = devtools::Devtools::new(Some(&window), &session.gpu, catalog, &assets, settings)?;
            session.set_overlay(Box::new(devtools));
        }
        if let Some(path) = pending.overlay_shot {
            session.overlay_shot_at_end(path);
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

/// The packaged game beside the executable, if there is one.
fn packaged_game() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let directory = exe.parent()?.join("game");
    directory.join("game.wasm").exists().then_some(directory)
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

const COMMANDS: [CommandSpec; 12] = [
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
    CommandSpec { name: "check", operands: "", checkout: true, about: "compile the game and the dev tools" },
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
        about: "golden-image tests (default tests/visual/scenarios.json)",
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
        name: "clean",
        operands: "",
        checkout: false,
        about: "delete pen's caches in the platform cache directory: compiled games, plugins and GPU \
                pipelines, and what plugins run and keep",
    },
];

/// Playing: a packaged game, `dev` and `run`.
const PLAY: &[&str] = &["", "dev", "run"];
/// Every command that works on a checkout.
const CHECKOUT: &[&str] =
    &["dev", "web", "run", "build", "check", "codegen", "export", "test", "bench", "packets", "test-blot"];

const OPTIONS: [OptionSpec; 43] = [
    OptionSpec { name: "--help", value: "", commands: &[], place: Place::Prose, about: "this" },
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
        commands: &["web"],
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
        about: "the runtime to ship (default player/ beside this executable, else this one); it must be \
                built from the same pen",
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
        name: "--web",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "the Rust browser host and the game as static files instead",
    },
    OptionSpec {
        name: "--check",
        value: "",
        commands: &["codegen"],
        place: Place::Command,
        about: "check the generated code instead of writing it",
    },
    OptionSpec {
        name: "--rust",
        value: "FILE",
        commands: &["codegen"],
        place: Place::Command,
        about: "where the Rust protocol code goes",
    },
    OptionSpec { name: "--out", value: "DIR", commands: &["export"], place: Place::Command, about: "where (required)" },
    OptionSpec {
        name: "--update",
        value: "",
        commands: &["test"],
        place: Place::Command,
        about: "write the shots as the golden images",
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
        commands: &["test"],
        place: Place::Command,
        about: "the golden images (default golden/ beside FILE)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["test"],
        place: Place::Command,
        about: "where shots and differences go (default .pen/visual)",
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
        about: "modules compiled at once, each worker in one compiler session (default half the cores, at most 8)",
    },
    OptionSpec { name: "--mute", value: "", commands: PLAY, place: Place::Play, about: "no sound" },
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
        commands: &["", "dev", "run", "bench", "packets"],
        place: Place::Play,
        about: "stop after N frames (bench: default 600; packets: default 120, or as long as the script lasts)",
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
        commands: &["", "dev", "run", "bench", "packets"],
        place: Place::Play,
        about: "scripted input: wait=N key=K down=K up=K type=T move=X,Y click=X,Y press=B release=B \
                drag=X,Y,X,Y wheel=D[,X,Y] shot=NAME save=F load=F quit; ui=ID,ACTION[,VALUE] (an assistive \
                technology's action on an interface node: focus, blur, activate, increment, decrement, \
                set_value, expand or collapse); tune=NAME=VALUE (a live-tunable constant of the checkout's \
                src/tunables.json); overlay=/COMMAND[,ARG...] (a line typed into the dev tools' console)",
    },
    OptionSpec {
        name: "--script-file",
        value: "FILE",
        commands: &["", "dev", "run", "bench", "packets"],
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
        name: "--game",
        value: "DIR",
        commands: &["", "test", "bench", "packets"],
        place: Place::Play,
        about: "a game bundle (`pen export`'s, or a package's game/) instead of the checkout's",
    },
    OptionSpec {
        name: "--assets",
        value: "DIR",
        commands: &["", "test", "bench", "packets"],
        place: Place::Play,
        about: "the bundle's assets from elsewhere",
    },
    OptionSpec {
        name: "--scene",
        value: "FILE",
        commands: &["", "test", "bench", "packets"],
        place: Place::Play,
        about: "the bundle's scene from elsewhere, which the editor saves to",
    },
    OptionSpec {
        name: "--software",
        value: "",
        commands: &["", "dev", "run", "test"],
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
        about: "keep the last 10 s to rewind and replay (always in dev and replays)",
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
];

/// Every variable of the environment pen reads, and what it does.
const ENVIRONMENT: [(&str, &str); 19] = [
    ("PEN_PRESENT_MODE=MODE", "auto-vsync (the default), auto-no-vsync, fifo, fifo-relaxed, immediate or mailbox"),
    ("PEN_SOFTWARE=1", "as --software"),
    ("PEN_OVERLAY=1", "as --overlay"),
    ("PEN_SMOKE_FRAMES=N", "as --frames N"),
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
    ("PEN_GPU_BACKTRACE=1", "where a GPU validation error came from"),
    ("PEN_SERVICES_LOG=1", "a line for each request the game makes of its services"),
    (
        "PEN_A11Y_DUMP=FILE",
        "the game's interface as assistive technology sees it, written to FILE whenever it changes (headless too)",
    ),
    ("PEN_RECORD_PCM=FILE", "a recording's sound as mixed, before it is encoded (f32 stereo at 48 kHz)"),
    ("PEN_STEAM=1", "Steam outside a package (pen built with Steam)"),
    ("PEN_STEAM_DEBUG=STEPS", "achievement and stat steps to run against Steam (see steam.rs)"),
    ("PEN_STEAM_APP_ID=ID", "relaunch through Steam when started outside it"),
    ("PEN_CRASH_TEST=N", "panic at frame N, to try crash reports"),
    ("PEN_PLUGINS=DIR", "where pen's own plugins are (default plugins/ beside pen)"),
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
            if !word.starts_with('-') {
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
        let most = if operands.is_empty() {
            0
        } else if operands.ends_with("...]") {
            usize::MAX
        } else {
            1
        };
        let least = usize::from(!operands.is_empty() && !operands.starts_with('['));
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
    let options = Options {
        frames: frames.or(smoke).unwrap_or(if shot && script.is_empty() { 120 } else { 0 }),
        screenshot,
        record,
        fps,
        mute: args.flag("--mute"),
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
    };
    if headless && options.frames == 0 && options.script.is_empty() && playback.replay.is_none() {
        bail!("--headless needs --frames N, a --script or a --replay to know when to stop");
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
    // Development recompiles after every edit, reusing unchanged functions.
    let engine = if dev { blot_host::dev_engine()? } else { blot_host::engine()? };
    let module = load_module(&engine, &game.directory, !dev)?;
    let catalog = read_catalog(&paths.assets)?;
    #[cfg(feature = "devtools")]
    let devtools = |keep: bool| {
        checkout.map(|root| devtools::Settings {
            project: root.to_owned(),
            build: game.directory.clone(),
            open: devtools::requested(args.flag("--overlay")),
            keep,
        })
    };

    if headless {
        let gpu = Gpu::headless(size.unwrap_or((1920, 1280)), &paths.assets, software)?;
        #[cfg(feature = "devtools")]
        let (assets, devtools_catalog) = (paths.assets.clone(), catalog.clone());
        let mut session = Session::start(options, paths, engine, module, catalog, gpu, None)?;
        session.playback(playback, None)?;
        #[cfg(feature = "devtools")]
        if let Some(settings) = devtools(false).filter(|_| overlay_asked) {
            let mut devtools = devtools::Devtools::new(None, &session.gpu, devtools_catalog, &assets, settings)?;
            // Headless, the program's build is waited for: what it shows is the point.
            devtools.start_now()?;
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
            #[cfg(feature = "devtools")]
            devtools: devtools(true),
            overlay_shot,
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

fn main() -> ExitCode {
    logger::install();
    match run(std::env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            crash::stopped(&error);
            ExitCode::FAILURE
        }
    }
}

fn run(words: impl IntoIterator<Item = String>) -> Result<()> {
    let args = Args::parse(words)?;
    if args.flag("--help") {
        println!("{}", usage());
        return Ok(());
    }
    if args.flag("--version") {
        println!("pen {} (game runtime: {})", env!("CARGO_PKG_VERSION"), project::RUNTIME);
        return Ok(());
    }
    if args.command == "clean" {
        return clean();
    }
    // The checkout, for its commands but those given a game bundle instead.
    let bundled =
        args.command.is_empty() || (["test", "bench", "packets"].contains(&args.command) && args.flag("--game"));
    let (project, bundle) = if bundled {
        let directory = bundle_directory(&args)?;
        let _ = IDENTITY.set(project::bundle_identity(&directory)?);
        (None, Some(directory))
    } else {
        let project = Project::find(args.path("--project"))?;
        let _ = IDENTITY.set(project.identity.clone());
        (Some(project), None)
    };
    crash::install(args.path("--data"));
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
            project.compile_devtools(&project.native())
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
            let project = checkout()?;
            if args.flag("--web") {
                let out = args.path("--out").unwrap_or_else(|| project.root.join(".pen/web-export"));
                return project::export_web(project, &out);
            }
            let out = args.path("--out").unwrap_or_else(|| project.root.join(".pen/package"));
            let options = project::PackageOptions {
                runtime: args.path("--runtime"),
                target: args.value("--target"),
                cpu: args.value("--cpu"),
            };
            project::package(project, &out, options)
        }
        "test" => {
            if let Some(project) = &project {
                project.compile(&project.native())?;
            }
            let game = game(project.as_ref());
            let root = match &project {
                Some(project) => project.root.clone(),
                None => std::env::current_dir()?,
            };
            let file =
                args.operands.first().map(PathBuf::from).unwrap_or_else(|| root.join("tests/visual/scenarios.json"));
            let engine = blot_host::engine()?;
            let module = load_module(&engine, &game.directory, true)?;
            let catalog = read_catalog(&game.assets)?;
            let test = visual::TestOptions {
                golden: args.path("--golden").unwrap_or_else(|| file.parent().unwrap_or(Path::new(".")).join("golden")),
                out: args.path("--out").unwrap_or_else(|| root.join(".pen/visual")),
                file,
                update: args.flag("--update"),
                only: args.values("--only"),
                software,
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
            let module = load_module(&engine, &game.directory, true)?;
            let frames: Option<usize> = args.number("--frames")?;
            let steps = args.script.clone();
            let out = (command == "packets").then(|| PathBuf::from(&args.operands[0]));
            let frames = match (command, steps.is_empty()) {
                ("bench", _) => Some(frames.unwrap_or(600)),
                (_, true) => Some(frames.unwrap_or(120)),
                _ => frames,
            };
            packets::run(packets::Run {
                module: &module,
                engine: &engine,
                assets: game.assets.clone(),
                scene: game.scene_to_start(),
                viewport: Viewport { width: 1280, height: 720 },
                out,
                frames,
                steps,
            })
            .map(drop)
        }
        #[cfg(feature = "compiler")]
        "test-blot" => {
            let project = checkout()?;
            let files = args.operands.iter().map(PathBuf::from).collect();
            let options = blot_test::Options { files, filter: args.value("--filter"), jobs: args.number("--jobs")? };
            blot_test::run(&project.root, options)
        }
        #[cfg(not(feature = "compiler"))]
        "test-blot" => bail!("this pen has no plugins, which compile a checkout's Blot tests"),
        other => unreachable!("Args::parse takes only known commands, not {other}"),
    }
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
        assert!(parse("--version").unwrap().flag("--version"));
        assert!(parse("clean").is_ok());
    }

    #[test]
    fn mistakes_are_refused_with_what_was_meant() {
        assert_eq!(
            error("tset"),
            "unknown command tset (did you mean test?); the commands are dev, web, run, build, check, codegen, export, test, bench, packets, test-blot and clean (pen --help)"
        );
        assert_eq!(error("run --frame 300"), "pen run has no option --frame (did you mean --frames?) (pen --help)");
        assert_eq!(error("run --port 8000"), "--port is for pen web, not pen run");
        assert_eq!(
            error("build --frames 3"),
            "--frames is for pen, pen dev, pen run, pen bench and pen packets, not pen build"
        );
        assert_eq!(error("--steam"), "pen has no option --steam (pen --help)");
        assert_eq!(error("dev --overlay=1"), "--overlay takes no value");
        assert_eq!(error("run --frames"), "--frames takes a value: --frames N");
        assert_eq!(error("build --out --web"), "--out takes a value: --out DIR");
        assert_eq!(error("run extra"), "pen run takes no operands, not extra");
        assert_eq!(error("test a.json b.json"), "pen test takes [FILE] and options, not b.json");
        assert_eq!(error("packets"), "pen packets needs DIR");
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
            "PEN_WEB_HOST",
            "PEN_STEAM_APP_ID",
            "PEN_CRASH_TEST",
            "BLOT_INLINING",
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
