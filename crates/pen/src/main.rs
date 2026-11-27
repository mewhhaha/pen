//! Native front end: windows, rendering, audio, headless verification and packaging.
// The player runtime opens no console window on Windows (an error it stops
// with goes to a crash report and a message box); the development one keeps
// its console.
#![cfg_attr(all(windows, not(feature = "compiler")), windows_subsystem = "windows")]
mod accessibility;
mod app_bundle;
#[cfg(feature = "compiler")]
mod asset_graph;
#[cfg(feature = "compiler")]
mod asset_lint;
#[cfg(feature = "compiler")]
mod atlas;
mod audio;
#[cfg(feature = "compiler")]
mod bake_lightmaps;
#[cfg(feature = "compiler")]
mod bench;
mod bugreport;
mod cli;
#[cfg(all(test, feature = "compiler"))]
mod codec_round_trip;
mod compiled;
mod crash;
mod delta;
#[cfg(feature = "compiler")]
mod dev;
mod devlog;
#[cfg(feature = "devtools")]
mod devtools;
#[cfg(feature = "compiler")]
mod diverge;
#[cfg(test)]
mod fixtures;
#[cfg(feature = "compiler")]
mod fuzz;
mod game;
mod gamepad;
#[cfg(feature = "compiler")]
mod gltf;
mod input;
mod kernels;
mod logger;
#[cfg(feature = "compiler")]
mod loudness;
#[cfg(feature = "devtools")]
mod materials;
mod media;
mod mods;
mod net;
#[cfg(feature = "compiler")]
mod new;
#[cfg(feature = "compiler")]
mod optimize;
mod opus;
#[cfg(feature = "compiler")]
mod packets;
mod patch;
#[cfg(feature = "compiler")]
mod pipeline;
#[cfg(feature = "compiler")]
mod plugin_build;
mod presets;
mod program;
mod project;
#[cfg(feature = "compiler")]
mod publish;
#[cfg(feature = "compiler")]
mod push;
mod record;
mod recording;
#[cfg(feature = "devtools")]
mod remote;
mod reports;
#[cfg(feature = "compiler")]
mod reproducible;
#[cfg(feature = "compiler")]
mod review;
mod saves;
mod scene_assets;
mod script;
mod services;
mod session;
mod snapshot;
mod spectate;
#[cfg(feature = "compiler")]
mod squid_test;
#[cfg(feature = "compiler")]
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
#[cfg(feature = "compiler")]
mod visual;
#[cfg(feature = "compiler")]
mod web;
#[cfg(feature = "compiler")]
mod web_export;
mod workshop;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use cli::{Args, load_mods, size, usage};
use pen_protocol::Catalog;
#[cfg(feature = "compiler")]
use pen_protocol::Viewport;
use pen_protocol::boundary::read_catalog;
use squid_host::GuestModule;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::project::Identity;
#[cfg(feature = "compiler")]
use crate::project::Project;
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
#[cfg(feature = "compiler")]
fn cache_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|directory| directory.join("pen"))
}

/// What pen keeps in its cache directory, which `pen clean` deletes:
/// compiled games, compiled plugins, what plugins run and keep, the GPU's
/// compiled pipelines, what asset steps built, and tests' scratch files.
#[cfg(feature = "compiler")]
const CACHES: [&str; 8] = ["wasm", "components", "programs", "plugins", "pipelines", "meshlets", "assets", "tmp"];

/// `pen clean`: deletes pen's caches, and nothing else there.
#[cfg(feature = "compiler")]
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

/// Shared compiler-free packaged worker loader called by BOTH packaged playback
/// (`play`) and `pen check-workers <package-dir>` (§5.1).
///
/// - Ordinary packages (`manifest.worker_deadlines == false`) reuse
///   `(game_engine, game_module, false)` directly from `game.cwasm`.
/// - Deadline-bounded packages (`manifest.worker_deadlines == true`) require
///   BOTH `workers.cwasm` and `workers.meta` on disk, instantiate compiler-free
///   `bounded_engine(2 ms)` with a live wakeable ticker, and deserialize via
///   `GuestModule::from_native`.
fn load_packaged_workers(
    game_dir: &Path,
    game_engine: &wasmtime::Engine,
    game_module: &GuestModule,
) -> Result<(wasmtime::Engine, GuestModule, bool)> {
    let Some(manifest) = project::Manifest::read(game_dir)? else {
        return Ok((game_engine.clone(), game_module.clone(), false));
    };
    if !manifest.worker_deadlines {
        return Ok((game_engine.clone(), game_module.clone(), false));
    }
    ensure!(
        manifest.runtime.as_deref() == Some(project::RUNTIME),
        "this native game was built for another runtime; package it again with `pen build`"
    );
    let workers_meta_path = game_dir.join("workers.meta");
    let workers_cwasm_path = game_dir.join("workers.cwasm");
    ensure!(
        workers_meta_path.is_file() && workers_cwasm_path.is_file(),
        "game.json sets worker_deadlines: true, but {} or {} is missing; repackage with `pen build --worker-deadlines`",
        workers_meta_path.display(),
        workers_cwasm_path.display()
    );
    let worker_engine = squid_host::bounded_engine(std::time::Duration::from_millis(2))?;
    let meta = std::fs::read(&workers_meta_path).with_context(|| format!("reading {}", workers_meta_path.display()))?;
    let compiled = compiled::decode(
        &std::fs::read(&workers_cwasm_path).with_context(|| format!("reading {}", workers_cwasm_path.display()))?,
    )?;
    // SAFETY: both files ship with this executable and are written by `pen build --worker-deadlines`.
    let worker_module = unsafe { GuestModule::from_native(&worker_engine, &meta, &compiled) }?;
    Ok((worker_engine, worker_module, true))
}

/// Compiler-free `pen check-workers <package-dir>` verification subcommand (§5.1).
fn check_workers(package_dir: &Path) -> Result<()> {
    let game_dir =
        if package_dir.join("game.meta").is_file() { package_dir.to_path_buf() } else { package_dir.join("game") };
    ensure!(
        game_dir.join("game.meta").is_file() && game_dir.join("game.cwasm").is_file(),
        "packaged game not found in {} (expected game.meta and game.cwasm)",
        game_dir.display()
    );
    let game_engine = squid_host::engine()?;
    let game_module = load_module(&game_engine, &game_dir, false, false)?;
    let (worker_engine, worker_module, has_deadlines) = load_packaged_workers(&game_dir, &game_engine, &game_module)?;
    if worker_module.abi().kernel("double_fixed").is_some() {
        let cfg = squid_host::kernels::KernelConfig {
            workers: 2,
            deadline: if has_deadlines { Some(std::time::Duration::from_millis(500)) } else { None },
            ..squid_host::kernels::KernelConfig::default()
        };
        let mut pool = squid_host::kernels::KernelPool::open(&worker_engine, &worker_module, "double_fixed", cfg)?;
        let out = pool.run_fixed(&[1, 2, 3, 4])?;
        ensure!(out == vec![2, 4, 6, 8], "double_fixed output mismatch: got {out:?}");
        if !has_deadlines {
            let bounded_cfg = squid_host::kernels::KernelConfig {
                workers: 1,
                deadline: Some(std::time::Duration::from_millis(50)),
                ..squid_host::kernels::KernelConfig::default()
            };
            let err =
                squid_host::kernels::KernelPool::open(&worker_engine, &worker_module, "double_fixed", bounded_cfg)
                    .err()
                    .context("ordinary package should reject deadline on non-epoch engine")?;
            ensure!(
                err == squid_host::kernels::KernelError::DeadlineRequiresBoundedEngine,
                "expected DeadlineRequiresBoundedEngine, got {err:?}"
            );
        }
    }
    if has_deadlines && worker_module.abi().kernel("spin_forever").is_some() {
        let spin_cfg = squid_host::kernels::KernelConfig {
            workers: 2,
            deadline: Some(std::time::Duration::from_millis(40)),
            ..squid_host::kernels::KernelConfig::default()
        };
        let mut pool = squid_host::kernels::KernelPool::open(&worker_engine, &worker_module, "spin_forever", spin_cfg)?;
        let err = pool.run_fixed(&[1, 2]).expect_err("spin_forever must time out on bounded worker engine");
        ensure!(err == squid_host::kernels::KernelError::Timeout, "expected Timeout for spin_forever, got {err:?}");
    }
    eprintln!("check-workers OK: worker_deadlines={has_deadlines}, kernels={}", worker_module.abi().kernels.len());
    Ok(())
}

/// The game's module: `game.cwasm` when the package precompiled it for this
/// runtime, else `game.wasm` compiled now (and cached).
fn load_module(engine: &wasmtime::Engine, game: &Path, cache: bool, measure_systems: bool) -> Result<GuestModule> {
    load_module_with(engine, game, cache, measure_systems, false)
}

/// Like [`load_module`], but prefers `wtime-aot` for `pen run` builds when
/// available (disable with `PEN_WTIME_AOT=0`).
fn load_run_module(engine: &wasmtime::Engine, game: &Path, cache: bool, measure_systems: bool) -> Result<GuestModule> {
    load_module_with(engine, game, cache, measure_systems, true)
}

fn load_module_with(
    engine: &wasmtime::Engine,
    game: &Path,
    cache: bool,
    measure_systems: bool,
    prefer_aot: bool,
) -> Result<GuestModule> {
    let metadata = game.join("game.meta");
    if metadata.is_file() {
        let started = Instant::now();
        let manifest = project::Manifest::read(game)?.context("native game is missing its manifest")?;
        ensure!(
            manifest.runtime.as_deref() == Some(project::RUNTIME),
            "this native game was built for another runtime; package it again with `pen build`"
        );
        let metadata = std::fs::read(&metadata)?;
        let compiled = compiled::decode(&std::fs::read(game.join("game.cwasm"))?)?;
        // SAFETY: both files ship with this executable and are written by `pen build`.
        let module = unsafe { GuestModule::from_native(engine, &metadata, &compiled) }?;
        eprintln!("game module ready in {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
        startup::mark("program loaded");
        return Ok(module);
    }
    #[cfg(not(feature = "compiler"))]
    {
        let _ = (cache, measure_systems, prefer_aot);
        bail!("{} is missing; package the game again with `pen build`", metadata.display())
    }
    #[cfg(feature = "compiler")]
    {
        let path = game.join("game.wasm");
        let wasm = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let cache = if cache { cache_dir().map(|directory| directory.join("wasm")) } else { None };
        let started = Instant::now();
        #[cfg(feature = "compiler")]
        if let Some(path) = std::env::var_os("PEN_TRACE_SYSTEMS") {
            let limit = std::env::var("PEN_TRACE_LIMIT")
                .ok()
                .map(|value| value.parse::<u64>())
                .transpose()?
                .unwrap_or(1_000_000);
            let trace = squid_host::SystemTrace::create(Path::new(&path), limit)?;
            eprintln!("system trace: {} (up to {limit} events; summary on exit)", Path::new(&path).display());
            return GuestModule::traced(engine, &wasm, cache.as_deref(), trace);
        }
        #[cfg(feature = "compiler")]
        if measure_systems && squid_host::read_abi(&wasm)?.function("main_trace").is_some() {
            return GuestModule::traced(engine, &wasm, cache.as_deref(), squid_host::SystemTrace::live());
        }
        let _ = measure_systems;
        let module = if prefer_aot {
            GuestModule::new_run(engine, &wasm, cache.as_deref())?
        } else {
            GuestModule::new(engine, &wasm, cache.as_deref())?
        };
        if module.backend_name() == "wtime-aot" {
            eprintln!("game module ready in {:.0} ms (wtime-aot)", started.elapsed().as_secs_f64() * 1000.0);
        } else {
            eprintln!("game module ready in {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
        }
        startup::mark("program loaded");
        Ok(module)
    }
}

/// What the window needs to start a session once it exists.
struct Pending {
    options: Options,
    playback: Playback,
    paths: Paths,
    engine: wasmtime::Engine,
    module: GuestModule,
    worker_engine: wasmtime::Engine,
    worker_module: GuestModule,
    worker_deadlines: bool,
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
        session.set_worker_module(pending.worker_engine, pending.worker_module, pending.worker_deadlines);
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
    #[cfg(feature = "compiler")]
    fn checkout(project: &Project) -> Game {
        Game { directory: project.native(), assets: project.assets(), scene: Some(project.scene()) }
    }

    #[cfg(feature = "compiler")]
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
    #[cfg(feature = "compiler")]
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
    #[cfg(not(feature = "compiler"))]
    bail!("there is no game/ folder beside this player; pass --game DIR or package with `pen build`")
}

/// Says which of a script's `tune=` steps name no tunable of the checkout's
/// `src/tunables.json`: the game would take them and change nothing.
#[cfg(feature = "compiler")]
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
    #[cfg(feature = "compiler")]
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
        net::check_players(identity().players, spec.players())?;
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
    let engine = if dev { squid_host::dev_engine()? } else { squid_host::engine()? };
    let module = if !dev {
        load_run_module(&engine, &game.directory, true, checkout.is_some())?
    } else {
        load_module(&engine, &game.directory, false, checkout.is_some())?
    };
    let (worker_engine, worker_module, worker_deadlines) = if game.directory.join("game.meta").is_file() {
        load_packaged_workers(&game.directory, &engine, &module)?
    } else {
        (engine.clone(), module.clone(), engine.get_epoch_interruption())
    };
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
        session.set_worker_module(worker_engine, worker_module, worker_deadlines);
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
            worker_engine,
            worker_module,
            worker_deadlines,
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
    let root = new::create(Path::new(&args.operands[0]), args.path("--squid"))?;
    println!(
        "made {} in {}: `pen dev` there plays it as you edit it, `pen test-squid` runs its tests",
        project::Project::find(Some(root.clone()))?.identity.name,
        root.display()
    );
    Ok(())
}

/// The commands that work over the network, which need no checkout: none
/// for another command.
#[cfg(feature = "compiler")]
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
        "relay" => Some((|| {
            let listen = args.value("--listen").unwrap_or_else(|| "0.0.0.0:7813".to_owned());
            let mut server =
                net::TcpRelayServer::bind(&listen).with_context(|| format!("binding relay on {listen}"))?;
            eprintln!("relay: listening for TCP and WebSocket netplay peers on {}", server.local_addr()?);
            server.serve(|| false)
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
#[cfg(feature = "compiler")]
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

/// Shipping entrypoint: playback and applying a package update. Development
/// commands are compiled only into the CLI.
#[cfg(not(feature = "compiler"))]
fn run(words: impl IntoIterator<Item = String>) -> Result<()> {
    let mut args = Args::parse(words)?;
    if args.flag("--help") {
        println!("{}", usage());
        return Ok(());
    }
    if args.flag("--version") {
        println!("pen {} (game runtime: {})", env!("CARGO_PKG_VERSION"), project::RUNTIME);
        return Ok(());
    }
    if args.command == "apply" {
        let package = args
            .operands
            .get(1)
            .map(PathBuf::from)
            .or_else(|| packaged_game().and_then(|game| game.parent().map(Path::to_owned)))
            .context("there is no package beside this executable: apply PATCH DIR")?;
        let version = patch::apply(Path::new(&args.operands[0]), &package)?;
        eprintln!("{} is now version {version}", package.display());
        return Ok(());
    }
    if args.command == "check-workers" {
        return check_workers(Path::new(&args.operands[0]));
    }
    ensure!(args.command.is_empty(), "{} is a development command; use the full pen CLI", args.command);
    let directory = bundle_directory(&args)?;
    let _ = IDENTITY.set(project::bundle_identity(&directory)?);
    script::name_game_events(&mut args.script, &|name, values| game_event(None, name, values))?;
    crash::install(args.path("--data"));
    if args.flag("--headless") {
        crash::unattended();
    }
    reports::install(&directory, args.path("--data").unwrap_or_else(data_dir));
    let software = args.flag("--software") || env_flag("PEN_SOFTWARE");
    play(&args, Game::bundle(&args, directory), false, packaged_game().is_some(), None, software)
}

#[cfg(feature = "compiler")]
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
    if args.command == "check-workers" {
        return check_workers(Path::new(&args.operands[0]));
    }
    if args.command == "gltf" {
        return import_gltf(&args);
    }
    if args.command == "import-scene" {
        return import_gltf_scene(&args);
    }
    if args.command == "bake-lightmaps" {
        let opts = bake_lightmaps::BakeCliOptions {
            job_path: PathBuf::from(&args.operands[0]),
            out_dir: args.path("--out").unwrap_or_else(|| PathBuf::from("lightmaps")),
            size_override: args.number::<u32>("--size")?,
            samples_override: args.number::<u32>("--samples")?,
            bounces_override: args.number::<u32>("--bounces")?,
        };
        return bake_lightmaps::run(&opts);
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
        let project = if args.command == "tool"
            || (matches!(args.command, "test-guest" | "test-squid") && !args.flag("--session"))
        {
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
        "codegen" => {
            let root = &checkout()?.root;
            let rust_output = args.path("--rust");
            let check = args.flag("--check");
            if let Some(plugins) = plugin_build::load(root)? {
                let extensions_json = pen_codegen::checkout_extensions_json(root)?;
                let generated = plugins.codegen(pen_codegen::SCHEMA, &extensions_json)?;
                let files: Vec<(String, String)> =
                    generated.files.into_iter().map(|file| (file.path, file.contents)).collect();
                pen_codegen::write_or_check(root, &files, &generated.remove, check)?;
                if let Some(path) = rust_output.as_deref() {
                    pen_codegen::generate_rust(path, check)?;
                }
                Ok(())
            } else {
                pen_codegen::generate(root, rust_output.as_deref(), check)
            }
        }
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
            let web = args.flag("--web");
            let compact = args.flag("--web-compact");
            let web_features = args.value("--web-features");
            ensure!(!compact || web, "--web-compact goes with --web");
            ensure!(web_features.is_none() || web, "--web-features goes with --web");
            let options = project::PackageOptions {
                optimize: !args.flag("--no-optimize"),
                runtime: args.path("--runtime"),
                target: args.value("--target"),
                cpu: args.value("--cpu"),
                bundle_id: args.value("--bundle-id"),
                store: args.path("--store"),
                worker_deadlines: args.flag("--worker-deadlines"),
            };
            // A bundle compiled already packages or exports to web as it is, with no compiler plugin needed.
            if let Some(bundle) = &bundle {
                if web {
                    let out = args.path("--out").context("pen build --web --game needs --out DIR")?;
                    return project::export_web(
                        project::GameSource::Bundle(bundle),
                        &out,
                        args.path("--zip").as_deref(),
                        options.optimize,
                        compact,
                        web_features.as_deref(),
                    );
                }
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
            let out = if web {
                let out = args.path("--out").unwrap_or_else(|| project.root.join(".pen/web-export"));
                project::export_web(
                    project::GameSource::Checkout(project),
                    &out,
                    args.path("--zip").as_deref(),
                    options.optimize,
                    compact,
                    web_features.as_deref(),
                )?;
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
            let engine = squid_host::engine()?;
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
            let engine = squid_host::engine()?;
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
            let mut results = bench::results(&[], load_ms, &measured)?;
            results.game = format!("{:016x}", project::game_fingerprint(&game.directory)?);
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
            let fuzzed =
                fuzz::Game::load(squid_host::engine()?, &game.directory, &game.assets, &game.scene_to_start())?;
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
        "test-guest" | "test-squid" => {
            let project = checkout()?;
            let files = args.operands.iter().map(PathBuf::from).collect();
            let deadline = args.number::<u64>("--deadline")?.map(std::time::Duration::from_secs);
            let session = if args.flag("--session") {
                ensure!(!args.operands.is_empty(), "--session requires explicit guest scenario files");
                Some(squid_test::SessionOptions {
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
                    "--out, --golden, --update and --software need --session for {}",
                    args.command
                );
                None
            };
            let options = squid_test::Options {
                files,
                filter: args.value("--filter"),
                jobs: args.number("--jobs")?,
                deadline,
                session,
            };
            squid_test::run(&project.root, options)
        }
        #[cfg(not(feature = "compiler"))]
        "test-guest" | "test-squid" => bail!("this pen has no plugins, which compile a checkout's guest tests"),
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
        ..Default::default()
    };
    let out = args.path("--out").unwrap_or_else(|| PathBuf::from("src/assets"));
    gltf::import(Path::new(&args.operands[0]), &out, &options)
}

/// `pen import-scene FILE`: full glTF scene/prefab import into assets + `.scene.json`.
#[cfg(feature = "compiler")]
fn import_gltf_scene(args: &Args) -> Result<()> {
    let options = pen_gltf::SceneImportOptions {
        name: args.value("--name"),
        shader: args.value("--shader").unwrap_or_else(|| "scene.wgsl".to_owned()),
        scene_path: args.path("--scene"),
        optimize_vertices: false,
        root_scale: args.number::<f32>("--scale")?.unwrap_or(1.0),
    };
    let out = args.path("--out").unwrap_or_else(|| PathBuf::from("src/assets"));
    let imported = pen_gltf::import_scene(Path::new(&args.operands[0]), &out, &options)?;
    println!(
        "Imported scene '{}' ({} nodes, {} meshes, {} LOD groups, {} colliders) -> {}",
        imported.scene.name,
        imported.scene.nodes.len(),
        imported.assets.meshes.len(),
        imported.lod_groups,
        imported.colliders,
        imported.scene_path.display()
    );
    Ok(())
}

#[cfg(all(test, not(feature = "compiler")))]
mod player_tests {
    use super::*;

    #[test]
    fn player_requires_current_native_metadata() {
        let root = project::temporary_dir().unwrap().join(format!("player-legacy-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("game.wasm"), b"\0asm\x01\0\0\0").unwrap();
        std::fs::write(root.join("game.cwasm"), b"old artifact must not be read").unwrap();
        let error = load_module(&squid_host::engine().unwrap(), &root, false, false).err().unwrap();
        assert!(error.to_string().contains("game.meta"), "{error:#}");
        assert!(error.to_string().contains("pen build"), "{error:#}");
        std::fs::remove_dir_all(root).unwrap();
    }
}
