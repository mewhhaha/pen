//! pen, gdev's native runtime: a winit window (or none), the wgpu renderer and the
//! game's Blot program on wasmtime. It does what gdev's TypeScript desktop
//! host does, and can also run headless, record, follow a script and hot
//! reload while developing.
// The player runtime opens no console window on Windows (crash reports go
// to a file); the development one keeps its console.
#![cfg_attr(all(windows, not(feature = "compiler")), windows_subsystem = "windows")]
mod accessibility;
mod audio;
mod blot_test;
mod crash;
mod dev;
#[cfg(feature = "devtools")]
mod devtools;
mod game;
mod input;
mod opus;
mod overlay;
mod packets;
mod parity;
mod pipeline;
mod program;
mod project;
mod record;
mod recording;
mod script;
mod services;
mod session;
mod snapshot;
#[cfg(feature = "steam")]
mod steam;
mod timeline;
mod visual;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use blot_host::GuestModule;
use pen_protocol::boundary::read_catalog;
use pen_protocol::{Catalog, Viewport};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::project::Project;
use crate::session::{Gpu, Options, Paths, Playback, Session};

/// The game this runtime plays, whose protocol it speaks: the name of its
/// data directory and of the executable `pen build` ships it as.
const GAME: &str = "gdev";

/// Where the game keeps saves, preferences and crash reports, unless
/// `--data DIR` says otherwise: the platform data directory's `gdev`.
fn data_dir() -> PathBuf {
    dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join(GAME)
}

fn cache_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|directory| directory.join("pen").join("wasm"))
}

/// The game's module: `game.cwasm` when the package precompiled it for this
/// machine, else `game.wasm` compiled now (and cached).
fn load_module(engine: &wasmtime::Engine, game: &Path, cache: bool) -> Result<GuestModule> {
    let path = game.join("game.wasm");
    let wasm = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let cache = if cache { cache_dir() } else { None };
    let started = Instant::now();
    let module = match std::fs::read(game.join("game.cwasm")) {
        // SAFETY: game.cwasm ships in the same package as this executable,
        // written by `pen build`.
        Ok(precompiled) => unsafe { GuestModule::with_precompiled(engine, &wasm, &precompiled, cache.as_deref())? },
        Err(_) if !blot_host::COMPILER => {
            bail!("{} is missing, and this runtime has no compiler: it plays games packaged by `pen build`", game.join("game.cwasm").display())
        }
        Err(_) => GuestModule::new(engine, &wasm, cache.as_deref())?,
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
        let mut attributes = Window::default_attributes().with_title(accessibility::TITLE).with_visible(false);
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

/// The packaged game beside the executable, else `dist/gdev` in this repo.
fn packaged_game() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let directory = exe.parent()?.join("game");
    directory.join("game.wasm").exists().then_some(directory)
}

const USAGE: &str = "\
usage: pen [COMMAND] [OPTIONS]

commands, in a gdev checkout (or with --project DIR):
  dev [--no-pipeline] compile the game's Blot program and play it, reloading the program,
                      shaders and assets as they change, with the game's state kept; and
                      rerun the checkout's asset recipes (pipeline.json) as their sources change
  web                 the checkout's browser build (`deno task web`): its console,
                      diagnostics and the browser's devtools, with hot reload
  run                 compile once and play
  build [--out DIR] [--runtime EXE] [--target TRIPLE] [--cpu PRESET]
                      a shippable folder: the player runtime (default: player/ beside this
                      executable, else this one), its libraries and the game, precompiled for
                      the runtime's platform and a CPU baseline (default x86-64-v2, or native)
                      (default .pen/package)
  build --web [--out DIR]
                      the browser build as static files for any web host (scripts/export_web.ts;
                      default .pen/web-export)
  test [FILE] [--update] [--only NAME...] [--golden DIR] [--out DIR]
                      golden-image tests (default tests/visual/scenarios.json)
  parity [--at N,N...] [--size WxH] [--script STEPS] [--threshold T]
                      compare the native renderer with renderer.ts on the same frames
  bench [--frames N]  guest time per frame, without rendering
  packets DIR [--frames N] [--script STEPS]
                      each frame's input and render packet, without rendering
  test-blot [FILE...] [--filter TEXT]
                      the checkout's Blot tests (default tests/*.blot): each module's test_ entries,
                      compiled against the game if its first line is `// test: game`, else alone

without a command, pen plays a packaged game: game/ beside the executable, or --game DIR.

options for playing (pen, dev, run):
  --mute              no sound
  --headless          no window; draw offscreen (steady clock, 60 fps unless --fps)
  --size WxH          the window's or the offscreen image's size in pixels
                      (headless default 1920x1280, where script coordinates match the browser's)
  --fps N             steady game clock: N frames per game second
  --frames N          stop after N frames
  --screenshot PNG    write the last frame
  --shots DIR         where shot=NAME steps write NAME.png (default: shots)
  --record VIDEO      record every frame (.mp4, .mkv or .webm, with sound unless --mute)
  --script STEPS      scripted input: wait=N key=K down=K up=K type=T move=X,Y click=X,Y
                      press=B release=B drag=X,Y,X,Y wheel=D[,X,Y] shot=NAME save=F load=F quit
                      ui=ID,ACTION[,VALUE] (an assistive technology's action on an interface node:
                      focus, blur, activate, increment, decrement, set_value, expand or collapse)
                      tune=NAME=VALUE (a live-tunable constant of the checkout's src/tunables.json)
                      overlay=/COMMAND[,ARG...] (a line typed into the dev tools' console)
  --script-file FILE  the same, from a file
  --data DIR          keep saves and preferences in DIR instead of the user's data directory
  --settle, --no-settle
                      draw the first frame until every asset has loaded before the clock starts
                      (default for --headless and --record)
  --game DIR, --assets DIR, --scene FILE
                      a game bundle, and assets or a scene from elsewhere
  --software          draw on a software GPU adapter (lavapipe, WARP), as PEN_SOFTWARE=1
  --record-input FILE record the session's input (a .penrec) to replay it exactly
  --replay FILE       play recorded input instead of the player's: the same frames, bit for bit
                      (headless, the run ends with it; in a window, it pauses there)
  --history           keep the last 10 s to rewind and replay (always in dev and replays)
  --overlap           compute the next frame while this one draws (one frame more input latency)
  --trace FILE        a line per frame: its number, and its input's and packet's hashes
  --overlay           open the dev tools' console from the start (dev and run; or PEN_OVERLAY=1)
  --overlay-shot PNG  write the last frame with the dev tools over it (headless too)

time travel: Ctrl+Shift+P pauses and plays, Ctrl+Shift+period steps a frame, Ctrl+Shift+comma
  steps back, Ctrl+Shift+R replays the last 5 s (after a reload, on the new build); in scripts,
  pause resume step[=N] rewind=N replay=SECONDS.

dev tools (dev and run): the checkout's own program for them (devtools/main.blot) runs beside
  the game and draws over it: its console on the backquote key (/help lists what it does), and
  its diagnostics; a checkout without one runs without dev tools.

environment: PEN_PRESENT_MODE, PEN_PROFILE_FRAMES, PEN_PROFILE_WARMUP, PEN_SMOKE_FRAMES, PEN_OVERLAY,
             PEN_GPU_BACKTRACE (where a GPU validation error came from), PEN_STEAM=1 (Steam
             outside a package), PEN_STEAM_DEBUG (achievement and stat steps, see steam.rs),
             PEN_SERVICES_LOG=1 (a line for each request the game makes of its services),
             PEN_SOFTWARE=1, BLOT_INLINING=0, PEN_A11Y_DUMP=FILE (the game's interface as assistive
             technology sees it, written to FILE whenever it changes; headless too)

a panic writes a crash report to crashes/ in the data directory (or --data DIR).";

fn size(text: &str) -> Result<(u32, u32)> {
    let (width, height) = text.split_once('x').context("--size takes WIDTHxHEIGHT")?;
    let size = (width.parse()?, height.parse()?);
    if size.0 == 0 || size.1 == 0 {
        bail!("--size must not be empty");
    }
    Ok(size)
}

/// Command-line arguments: a command, then `--name value` options and flags.
struct Args {
    command: Option<String>,
    positional: Vec<String>,
    args: Vec<String>,
}

impl Args {
    fn parse() -> Args {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let commands = ["dev", "web", "run", "build", "test", "parity", "bench", "packets", "test-blot"];
        let command = args.first().filter(|first| commands.contains(&first.as_str())).cloned();
        let rest = if command.is_some() { args[1..].to_vec() } else { args };
        // Words that are neither options nor their values.
        let mut positional = Vec::new();
        let mut index = 0;
        while index < rest.len() {
            let word = &rest[index];
            if word.starts_with("--") {
                index += if Args::takes_value(word) { 2 } else { 1 };
            } else {
                positional.push(word.clone());
                index += 1;
            }
        }
        Args { command, positional, args: rest }
    }

    fn takes_value(name: &str) -> bool {
        !matches!(
            name,
            "--mute"
                | "--headless"
                | "--settle"
                | "--no-settle"
                | "--update"
                | "--help"
                | "--version"
                | "--steam"
                | "--software"
                | "--no-pipeline"
                | "--web"
                | "--history"
                | "--overlap"
                | "--overlay"
        )
    }

    fn flag(&self, name: &str) -> bool {
        self.args.iter().any(|arg| arg == name)
    }

    fn values(&self, name: &str) -> Vec<String> {
        self.args
            .iter()
            .enumerate()
            .filter(|(_, arg)| *arg == name)
            .filter_map(|(index, _)| self.args.get(index + 1).cloned())
            .collect()
    }

    fn value(&self, name: &str) -> Option<String> {
        self.values(name).pop()
    }

    fn path(&self, name: &str) -> Option<PathBuf> {
        self.value(name).map(PathBuf::from)
    }

    fn script(&self) -> Result<Vec<script::Step>> {
        let mut words = self.values("--script");
        for file in self.values("--script-file") {
            let text = std::fs::read_to_string(&file).with_context(|| format!("reading {file}"))?;
            words.extend(text.lines().filter(|line| !line.trim_start().starts_with('#')).map(str::to_owned));
        }
        script::parse(&words)
    }
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
    fn bundle(args: &Args) -> Game {
        let directory = args.path("--game").or_else(packaged_game).unwrap_or_else(|| PathBuf::from("dist/gdev"));
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
    match overlay::load_tunables(root) {
        Ok(declared) => {
            for name in names.into_iter().filter(|name| declared.iter().all(|tunable| tunable.name != *name)) {
                eprintln!("script: tune={name} names no tunable in src/tunables.json, so it changes nothing");
            }
        }
        Err(problem) => eprintln!("script: {problem}"),
    }
}

/// Whether `run` builds the dev tools: a window always has them, a headless
/// run when its console should show, a shot of them is wanted or its script
/// types into them.
fn devtools_wanted(args: &Args) -> Result<bool> {
    Ok(!args.flag("--headless")
        || args.flag("--overlay")
        || args.path("--overlay-shot").is_some()
        || args.script()?.iter().any(|step| matches!(step, script::Step::Overlay(_))))
}

/// `checkout` is the gdev checkout played (`dev` and `run`), which has the
/// dev tools.
fn play(args: &Args, game: Game, dev: bool, steam: bool, checkout: Option<&Path>) -> Result<()> {
    let headless = args.flag("--headless");
    let frames: Option<usize> = args.value("--frames").map(|frames| frames.parse()).transpose()?;
    let smoke = std::env::var("PEN_SMOKE_FRAMES").ok().map(|frames| frames.parse()).transpose()?;
    let screenshot = args.path("--screenshot");
    let overlay_shot = args.path("--overlay-shot");
    let record = args.path("--record");
    let script = args.script()?;
    if let Some(root) = checkout {
        check_tunes(&script, root);
    }
    let settle = (headless || record.is_some() || args.flag("--settle")) && !args.flag("--no-settle");
    // A window always has the overlay; a headless run, when something asks for it.
    let overlay_asked = args.flag("--overlay")
        || overlay_shot.is_some()
        || script.iter().any(|step| matches!(step, script::Step::Overlay(_)));
    if overlay_asked && checkout.is_none() {
        bail!("the dev tools are for a checkout, in `pen dev` and `pen run`");
    }
    #[cfg(not(feature = "devtools"))]
    if overlay_asked {
        bail!("this pen has no dev tools (it was built without the devtools feature)");
    }
    let shot = screenshot.is_some() || overlay_shot.is_some();
    let options = Options {
        frames: frames.or(smoke).unwrap_or(if shot && script.is_empty() { 120 } else { 0 }),
        screenshot,
        record,
        fps: args.value("--fps").map(|fps| fps.parse()).transpose()?,
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
        pipeline: args.flag("--overlap"),
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
        let gpu = Gpu::headless(size.unwrap_or((1920, 1280)), &paths.assets)?;
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

fn main() -> Result<()> {
    let args = Args::parse();
    if args.flag("--help") || args.flag("-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args.flag("--version") {
        println!("pen {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    crash::install(args.path("--data"));
    if args.flag("--software") {
        // SAFETY: no other thread exists yet. The variable reaches every GPU
        // this process opens.
        unsafe { std::env::set_var("PEN_SOFTWARE", "1") };
    }
    let checkout = || Project::find(args.path("--project"));
    match args.command.as_deref() {
        None => play(&args, Game::bundle(&args), false, packaged_game().is_some(), None),
        Some("dev") => {
            let project = checkout()?;
            // The dev tools' program builds beside the game's.
            let _devtools = project.build_devtools(&project.native(), true)?;
            let _compiler = project.watch(&project.native())?;
            let _pipeline = (!args.flag("--no-pipeline")).then(|| pipeline::Pipeline::start(&project.root)).flatten();
            play(&args, Game::checkout(&project), true, false, Some(&project.root))
        }
        Some("web") => {
            let project = checkout()?;
            let mut command = std::process::Command::new("deno");
            command.args(["task", "web"]).current_dir(&project.root);
            // Become the server, so stopping pen stops it.
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                Err(command.exec()).context("running deno task web in the checkout")
            }
            #[cfg(not(unix))]
            {
                let status = command.status().context("running deno task web in the checkout")?;
                std::process::exit(status.code().unwrap_or(1));
            }
        }
        Some("run") => {
            let project = checkout()?;
            let mut devtools = if devtools_wanted(&args)? { project.build_devtools(&project.native(), false)? } else { None };
            project.compile(&project.native())?;
            // Headless, the dev tools are wanted from the first frame.
            if args.flag("--headless")
                && let Some(build) = &mut devtools
                && !build.wait()?
            {
                eprintln!("devtools: its program did not compile");
            }
            play(&args, Game::checkout(&project), false, false, Some(&project.root))
        }
        Some("build") => {
            let project = checkout()?;
            if args.flag("--web") {
                let out = args.path("--out").unwrap_or_else(|| project.root.join(".pen/web-export"));
                return project::export_web(&project, &out);
            }
            let out = args.path("--out").unwrap_or_else(|| project.root.join(".pen/package"));
            let options = project::PackageOptions {
                runtime: args.path("--runtime"),
                target: args.value("--target"),
                cpu: args.value("--cpu"),
            };
            project::package(&project, &out, options)
        }
        Some("test") => {
            let (game, root) = match args.path("--game") {
                Some(_) => (Game::bundle(&args), std::env::current_dir()?),
                None => {
                    let project = checkout()?;
                    project.compile(&project.native())?;
                    (Game::checkout(&project), project.root)
                }
            };
            let file = args
                .positional
                .first()
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("tests/visual/scenarios.json"));
            let engine = blot_host::engine()?;
            let module = load_module(&engine, &game.directory, true)?;
            let catalog = read_catalog(&game.assets)?;
            let test = visual::TestOptions {
                golden: args
                    .path("--golden")
                    .unwrap_or_else(|| file.parent().unwrap_or(Path::new(".")).join("golden")),
                out: args.path("--out").unwrap_or_else(|| root.join(".pen/visual")),
                file,
                update: args.flag("--update"),
                only: args.values("--only"),
            };
            let scene = game.scene_to_start();
            let visual_game = visual::Game { directory: &game.directory, assets: &game.assets, scene: &scene };
            visual::run(test, visual_game, engine, module, catalog)
        }
        Some("parity") => {
            let project = checkout()?;
            project.compile(&project.native())?;
            let engine = blot_host::engine()?;
            let module = load_module(&engine, &project.native(), true)?;
            let steps = args.script()?;
            let options = parity::ParityOptions {
                size: args.value("--size").map(|text| size(&text)).transpose()?.unwrap_or((960, 640)),
                steps: if steps.is_empty() {
                    script::parse(&[
                        "wait=30 key=Tab wait=40 click=470,150 wait=20 drag=300,500,600,450 wait=30 key=Tab wait=30".into(),
                    ])?
                } else {
                    steps
                },
                at: args
                    .value("--at")
                    .map(|list| list.split(',').map(str::parse).collect::<Result<Vec<usize>, _>>())
                    .transpose()?
                    .unwrap_or_default(),
                threshold: args.value("--threshold").map(|value| value.parse()).transpose()?.unwrap_or(0.002),
                out: args.path("--out").unwrap_or_else(|| project.root.join(".pen/parity")),
            };
            parity::run(&project, &engine, &module, options)
        }
        Some(command @ ("bench" | "packets")) => {
            let game = match args.path("--game") {
                Some(_) => Game::bundle(&args),
                None => {
                    let project = checkout()?;
                    project.compile(&project.native())?;
                    Game::checkout(&project)
                }
            };
            let engine = blot_host::engine()?;
            let module = load_module(&engine, &game.directory, true)?;
            let frames: Option<usize> = args.value("--frames").map(|frames| frames.parse()).transpose()?;
            let steps = args.script()?;
            let out = if command == "packets" {
                Some(PathBuf::from(args.positional.first().context("packets needs an output directory")?))
            } else {
                None
            };
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
        Some("test-blot") => {
            let project = checkout()?;
            let files = args.positional.iter().map(PathBuf::from).collect();
            blot_test::run(&project.root, blot_test::Options { files, filter: args.value("--filter") })
        }
        Some(other) => bail!("unknown command {other}"),
    }
}
