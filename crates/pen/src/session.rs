//! One running game: its GPU, renderer, program, audio and input, and the
//! frame loop the window and the headless runner share. Frames can also be
//! captured (screenshots, recordings), scripted, timed on a steady clock and
//! hot reloaded.
//!
//! The game is deterministic given its input, so a session can also record
//! its input and replay it exactly, pause, step, rewind to any recent frame
//! and play the last seconds again on a new build (see `timeline`), and
//! overlap frames: the guest computes the next frame while this one draws
//! (`Playback::pipeline`, one frame more input latency).
//!
//! In a window, the dev tools (`Overlay`) can draw over the game's frames.
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use blot_host::GuestModule;
use pen_protocol::{Catalog, GameEvent, RenderFrame, UiNode, Viewport};
use pen_render::RenderTarget;
use winit::event::WindowEvent;
use winit::window::Window;

use crate::audio::Audio;
use crate::dev::{Dev, Update};
use crate::game::{FrameTimings, GameRuntime, Host, Quiet, ReloadResult, hash_lanes};
use crate::input::{Hotkey, Input};
use crate::overlay::{self, Source};
use crate::record::Recorder;
use crate::recording;
use crate::script::{self, Script, Step, Tick, Travel};
use crate::services::Services;
use crate::snapshot::{self, SavedState};
use crate::timeline::{self, KEEP_SECONDS, Load, Recorded, Timeline};

/// Where the game's files are and where the player's are kept.
pub struct Paths {
    pub game: PathBuf,
    pub assets: PathBuf,
    pub quicksave: PathBuf,
    /// The scene the editor saves: the player's own, or the given file.
    pub scene: PathBuf,
    pub bundled_scene: PathBuf,
    pub preferences: PathBuf,
    /// Where `shot=NAME` steps write their PNGs.
    pub shots: PathBuf,
}

impl Paths {
    /// `data` replaces the platform data directory, where saves and
    /// preferences live (tests use a fresh one).
    pub fn new(
        game: PathBuf,
        assets: Option<PathBuf>,
        scene: Option<PathBuf>,
        shots: Option<PathBuf>,
        data: Option<PathBuf>,
    ) -> Paths {
        let data = data.unwrap_or_else(crate::data_dir);
        Paths {
            assets: assets.unwrap_or_else(|| game.join("assets")),
            bundled_scene: scene.clone().unwrap_or_else(|| game.join("main.scene")),
            scene: scene.unwrap_or_else(|| data.join("main.scene")),
            quicksave: data.join("quicksave.bin"),
            preferences: data.join("preferences.json"),
            shots: shots.unwrap_or_else(|| PathBuf::from("shots")),
            game,
        }
    }

    pub fn scene_to_read(&self) -> &Path {
        if self.scene.exists() { &self.scene } else { &self.bundled_scene }
    }
}

/// The game's clock: real time, or a steady `1/fps` per frame however long
/// frames take, for recordings and repeatable headless runs. Paused, real
/// time stops counting, and a frame stepped is `STEP_SECONDS` long.
pub struct Clock {
    kind: ClockKind,
    /// The last frame's time.
    last: f64,
    /// Real time paused at `last`.
    frozen: bool,
    /// Real time not counted, while paused.
    skipped: f64,
}

enum ClockKind {
    Real(Instant),
    Steady { fps: f64, frame: u64 },
}

/// The game time a frame stepped while paused takes on a real clock.
const STEP_SECONDS: f64 = 1.0 / 60.0;

impl Clock {
    fn real() -> Clock {
        Clock { kind: ClockKind::Real(Instant::now()), last: 0.0, frozen: false, skipped: 0.0 }
    }

    fn steady(fps: f64) -> Clock {
        Clock { kind: ClockKind::Steady { fps, frame: 0 }, last: 0.0, frozen: false, skipped: 0.0 }
    }

    fn is_steady(&self) -> bool {
        matches!(self.kind, ClockKind::Steady { .. })
    }

    fn next(&mut self) -> f64 {
        let timestamp = match &mut self.kind {
            ClockKind::Real(_) if self.frozen => self.last + STEP_SECONDS,
            ClockKind::Real(started) => started.elapsed().as_secs_f64() - self.skipped,
            ClockKind::Steady { fps, frame } => {
                let timestamp = *frame as f64 / *fps;
                *frame += 1;
                timestamp
            }
        };
        self.last = timestamp;
        timestamp
    }

    /// The time now, for the next frame's elapsed time to count from.
    fn now(&self) -> f64 {
        match &self.kind {
            ClockKind::Real(started) if !self.frozen => started.elapsed().as_secs_f64() - self.skipped,
            _ => self.last,
        }
    }

    fn pause(&mut self) {
        self.frozen = true;
    }

    fn resume(&mut self) {
        if let ClockKind::Real(started) = &self.kind
            && self.frozen
        {
            self.skipped = started.elapsed().as_secs_f64() - self.last;
        }
        self.frozen = false;
    }
}

/// The player's preferences, as the game last sent them. The game keeps
/// them; the host only stores them and plays them back as Preference events
/// at startup and after every restore.
pub struct Preferences {
    path: PathBuf,
    values: Vec<f32>,
}

impl Preferences {
    pub fn load(path: PathBuf) -> Preferences {
        let values = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Vec<f32>>(&text).ok())
            .unwrap_or_default();
        Preferences { path, values }
    }

    fn replay(&self, input: &mut Input) {
        for (index, value) in self.values.iter().enumerate() {
            input.enqueue(GameEvent::Preference { index: index as f32, value: *value });
        }
    }

    fn store(&mut self, values: &[f32]) -> Result<()> {
        if values == self.values {
            return Ok(());
        }
        self.values = values.to_vec();
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, serde_json::to_string(values)?)?;
        Ok(())
    }
}

/// What the last frame drawn held, for the diagnostics.
#[derive(Clone, Copy, Default)]
pub struct Drawn {
    pub meshes: usize,
    pub brushes: usize,
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub renderer: pen_render::Renderer,
    pub format: wgpu::TextureFormat,
    pub size: (u32, u32),
    pub drawn: Drawn,
    /// The window's swapchain; none when headless.
    pub surface: Option<(wgpu::Surface<'static>, wgpu::SurfaceConfiguration)>,
    /// Shows frames drawn offscreen in the window, scaled to fit.
    blitter: Option<wgpu::util::TextureBlitter>,
    /// A frame size the window does not change (a recording's).
    pinned: Option<(u32, u32)>,
    /// The offscreen image frames are captured from.
    capture: Option<wgpu::Texture>,
    fault: Fault,
}

/// An uncaptured GPU error or a lost device, which ends the run as gdev's
/// desktop host does.
type Fault = Arc<Mutex<Option<String>>>;

/// The adapter to draw with: the fast one, or with PEN_SOFTWARE=1 (the
/// `--software` flag) a software one, such as Mesa's lavapipe on Vulkan or
/// WARP on DX12, as on CI machines without a GPU.
fn request_adapter(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<wgpu::Adapter> {
    let software = std::env::var_os("PEN_SOFTWARE").is_some_and(|value| value != "0");
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: software,
        compatible_surface: surface,
        ..Default::default()
    }));
    if software {
        adapter.context("no software GPU adapter for PEN_SOFTWARE (on Linux: Mesa's lavapipe, in mesa-vulkan-drivers)")
    } else {
        adapter.context("no GPU adapter")
    }
}

fn open_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue, Fault)> {
    let info = adapter.get_info();
    eprintln!("GPU: {} ({:?})", info.name, info.backend);
    crate::crash::gpu(&info);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    let fault = Fault::default();
    let uncaptured = fault.clone();
    device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
        let trace = if std::env::var_os("PEN_GPU_BACKTRACE").is_some() {
            format!("\n{}", std::backtrace::Backtrace::force_capture())
        } else {
            String::new()
        };
        uncaptured.lock().unwrap().get_or_insert_with(|| format!("GPU error: {error}{trace}"));
    }));
    let lost = fault.clone();
    device.set_device_lost_callback(move |reason, message| {
        if reason != wgpu::DeviceLostReason::Destroyed {
            lost.lock().unwrap().get_or_insert_with(|| format!("GPU device lost: {message}"));
        }
    });
    Ok((device, queue, fault))
}

fn renderer(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, assets: &Path) -> Result<pen_render::Renderer> {
    pen_render::Renderer::new(
        device.clone(),
        queue.clone(),
        pen_render::RendererOptions { format, assets: assets.to_owned(), veil: Some(0.6) },
    )
}

impl Gpu {
    /// With `pinned`, frames keep that size whatever the window's.
    pub fn windowed(
        display: Box<dyn wgpu::wgt::WgpuHasDisplayHandle>,
        window: Arc<Window>,
        present_mode: wgpu::PresentMode,
        assets: &Path,
        pinned: Option<(u32, u32)>,
    ) -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(display));
        let surface = instance.create_surface(window.clone())?;
        let adapter = request_adapter(&instance, Some(&surface))?;
        let (device, queue, fault) = open_device(&adapter)?;
        let capabilities = surface.get_capabilities(&adapter);
        // Like the browser's preferred canvas format: 8 bits and not sRGB,
        // since the present pass encodes colours itself.
        let format = [wgpu::TextureFormat::Bgra8Unorm, wgpu::TextureFormat::Rgba8Unorm]
            .into_iter()
            .find(|format| capabilities.formats.contains(format))
            .or_else(|| capabilities.formats.iter().copied().find(|format| !format.is_srgb()))
            .unwrap_or_else(|| capabilities.formats[0].remove_srgb_suffix());
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: if capabilities.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
                wgpu::CompositeAlphaMode::Opaque
            } else {
                capabilities.alpha_modes[0]
            },
            view_formats: vec![],
            color_space: Default::default(),
        };
        surface.configure(&device, &config);
        let renderer = renderer(&device, &queue, format, assets)?;
        let blitter = wgpu::util::TextureBlitterBuilder::new(&device, format)
            .sample_type(wgpu::FilterMode::Linear)
            .build();
        Ok(Gpu {
            size: pinned.unwrap_or((config.width, config.height)),
            surface: Some((surface, config)),
            blitter: Some(blitter),
            pinned,
            device,
            queue,
            renderer,
            format,
            capture: None,
            drawn: Drawn::default(),
            fault,
        })
    }

    /// No window: every frame draws offscreen at `size`.
    pub fn headless(size: (u32, u32), assets: &Path) -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = request_adapter(&instance, None)?;
        let (device, queue, fault) = open_device(&adapter)?;
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let renderer = renderer(&device, &queue, format, assets)?;
        Ok(Gpu {
            device,
            queue,
            renderer,
            format,
            size,
            surface: None,
            blitter: None,
            pinned: Some(size),
            capture: None,
            drawn: Drawn::default(),
            fault,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.size = self.pinned.unwrap_or((width, height));
        if let Some((surface, config)) = &mut self.surface {
            config.width = width;
            config.height = height;
            surface.configure(&self.device, config);
        }
    }

    fn take_fault(&self) -> Option<String> {
        self.fault.lock().unwrap().take()
    }

    fn capture_texture(&mut self) -> wgpu::Texture {
        let (width, height) = self.size;
        if let Some(texture) = &self.capture
            && (texture.width(), texture.height()) == (width, height)
        {
            return texture.clone();
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.capture = Some(texture.clone());
        texture
    }

    /// The texture's pixels as BGRA (or RGBA for an RGBA format), rows packed.
    fn read(&self, texture: &wgpu::Texture) -> Result<Vec<u8>> {
        let (width, height) = (texture.width(), texture.height());
        let row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture readback"),
            size: u64::from(row * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(height) },
            },
            texture.size(),
        );
        self.queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        let mapped = slice.get_mapped_range().map_err(|error| anyhow!("reading a frame back: {error:?}"))?;
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let start = (y * row) as usize;
            pixels.extend_from_slice(&mapped[start..start + (width * 4) as usize]);
        }
        Ok(pixels)
    }

    /// Draws `frame` offscreen until its assets have loaded and settled
    /// (at least twice), and returns it as RGBA, rows packed.
    pub fn draw_settled(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<Vec<u8>> {
        const PATIENCE: std::time::Duration = std::time::Duration::from_secs(60);
        let started = Instant::now();
        let texture = self.capture_texture();
        for draws in 0.. {
            let output = Output {
                device: &self.device,
                queue: &self.queue,
                surface: None,
                size: self.size,
                blitter: None,
                overlay: None,
            };
            self.renderer.submit(frame, catalog, &mut CaptureTarget { output, texture: texture.clone() })?;
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
            if let Some(fault) = self.take_fault() {
                return Err(anyhow!(fault));
            }
            if (draws >= 1 && self.renderer.quiet()) || started.elapsed() > PATIENCE {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
        let pixels = self.read(&texture)?;
        Ok(if self.bgra() { swap_red_blue(pixels) } else { pixels })
    }

    fn bgra(&self) -> bool {
        matches!(self.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb)
    }
}

/// The dev tools over the window (`crate::devtools`): they see the window's
/// events before the game, run before each step (acting on the session),
/// and draw over the window's image after the game's frame and before it is
/// presented. They never draw into captures, recordings or headless output,
/// only into overlay shots.
pub trait Overlay {
    /// A window event, before the game: whether the game gets it too.
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool;
    /// Before a step: what the overlay shows, and what it asks of the
    /// session; headless, it has no window and lays itself out on the frame.
    fn update(&mut self, session: &mut Session, window: Option<&Window>);
    /// A line for its console, as if typed there (a script's `overlay=`),
    /// run on the next update.
    fn command(&mut self, _line: String) {}
    /// Draws what was laid out last over `target`, an image about to be shown.
    fn paint(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture);
    /// The pointer is the overlay's (over a panel, or dragging one), so it
    /// shows even where the game hides it.
    fn wants_pointer(&self) -> bool;
    /// It shows, with a tree of its own for assistive technology
    /// (`accessibility`).
    fn accesskit_shown(&self) -> bool {
        false
    }
    /// Its AccessKit tree from its last layout, once: a whole tree in its
    /// own ids (its own root tree).
    fn accesskit_update(&mut self) -> Option<accesskit::TreeUpdate> {
        None
    }
    /// An assistive technology's request for one of its nodes (in its own
    /// tree's terms), for its next layout.
    fn accesskit_action(&mut self, _request: accesskit::ActionRequest) {}
}

/// What a frame is presented to: the GPU fields a target needs, borrowed
/// apart from the renderer that draws into it.
struct Output<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    surface: Option<&'a (wgpu::Surface<'static>, wgpu::SurfaceConfiguration)>,
    size: (u32, u32),
    blitter: Option<&'a wgpu::util::TextureBlitter>,
    /// Draws over the swapchain image before it is presented.
    overlay: Option<&'a mut Box<dyn Overlay>>,
}

impl Output<'_> {
    fn acquire(&self) -> Option<wgpu::SurfaceTexture> {
        let (surface, config) = self.surface?;
        match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                Some(texture)
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                surface.configure(self.device, config);
                None
            }
            _ => None,
        }
    }
}

/// The window's swapchain image for one frame.
struct SurfaceTarget<'a> {
    output: Output<'a>,
    texture: Option<wgpu::SurfaceTexture>,
}

impl RenderTarget for SurfaceTarget<'_> {
    fn size(&self) -> (u32, u32) {
        self.output.size
    }

    fn view(&mut self) -> Option<wgpu::TextureView> {
        let texture = self.output.acquire()?;
        let view = texture.texture.create_view(&Default::default());
        self.texture = Some(texture);
        Some(view)
    }

    fn present(&mut self) {
        if let Some(texture) = self.texture.take() {
            if let Some(overlay) = self.output.overlay.as_deref_mut() {
                overlay.paint(self.output.device, self.output.queue, &texture.texture);
            }
            self.output.queue.present(texture);
        }
    }
}

/// An offscreen image to capture; a windowed run also copies it to the
/// swapchain so the window shows what is captured.
struct CaptureTarget<'a> {
    output: Output<'a>,
    texture: wgpu::Texture,
}

impl RenderTarget for CaptureTarget<'_> {
    fn size(&self) -> (u32, u32) {
        (self.texture.width(), self.texture.height())
    }

    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.texture.create_view(&Default::default()))
    }

    fn present(&mut self) {
        let Some(blitter) = self.output.blitter else { return };
        let Some(swapchain) = self.output.acquire() else { return };
        let mut encoder = self.output.device.create_command_encoder(&Default::default());
        blitter.copy(
            self.output.device,
            &mut encoder,
            &self.texture.create_view(&Default::default()),
            &swapchain.texture.create_view(&Default::default()),
        );
        self.output.queue.submit([encoder.finish()]);
        // Over the window's copy only: what is captured stays the game's.
        if let Some(overlay) = self.output.overlay.as_deref_mut() {
            overlay.paint(self.output.device, self.output.queue, &swapchain.texture);
        }
        self.output.queue.present(swapchain);
    }
}

/// What to keep of the frames drawn during one step.
#[derive(Default)]
struct Capture {
    screenshots: Vec<PathBuf>,
    recorder: Option<Recorder>,
    /// Always draw offscreen (headless).
    offscreen: bool,
    /// Screenshots with the overlay, as the window shows it.
    overlay_shots: Vec<PathBuf>,
}

/// Everything the game runtime may touch while a frame runs.
struct Shell<'a> {
    gpu: &'a mut Gpu,
    window: Option<&'a Window>,
    audio: Option<&'a mut Audio>,
    input: &'a mut Input,
    preferences: &'a mut Preferences,
    services: &'a mut Services,
    capture: &'a mut Capture,
    overlay: Option<&'a mut Box<dyn Overlay>>,
}

impl Host for Shell<'_> {
    fn viewport(&self) -> Viewport {
        Viewport { width: self.gpu.size.0.max(1), height: self.gpu.size.1.max(1) }
    }

    fn validate(&self, frame: &RenderFrame) -> Result<()> {
        pen_render::validate_frame(frame)
    }

    fn render(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<()> {
        let capture = &mut *self.capture;
        let offscreen = capture.offscreen
            || !capture.screenshots.is_empty()
            || capture.recorder.is_some()
            || !capture.overlay_shots.is_empty();
        let texture = offscreen.then(|| self.gpu.capture_texture());
        let gpu = &mut *self.gpu;
        gpu.drawn = Drawn { meshes: frame.draws.len(), brushes: frame.solids.len() };
        let output = Output {
            device: &gpu.device,
            queue: &gpu.queue,
            surface: gpu.surface.as_ref(),
            size: gpu.size,
            blitter: gpu.blitter.as_ref(),
            overlay: self.overlay.as_deref_mut(),
        };
        match &texture {
            Some(texture) => {
                let mut target = CaptureTarget { output, texture: texture.clone() };
                gpu.renderer.submit(frame, catalog, &mut target)?;
            }
            None => {
                let mut target = SurfaceTarget { output, texture: None };
                gpu.renderer.submit(frame, catalog, &mut target)?;
                target.present();
            }
        }
        if let Some(texture) = &texture {
            for path in capture.overlay_shots.drain(..) {
                write_overlay_shot(gpu, texture, self.overlay.as_deref_mut(), &path)?;
            }
        }
        if let Some(texture) = texture
            && (!capture.screenshots.is_empty() || capture.recorder.is_some())
        {
            let pixels = gpu.read(&texture)?;
            let (width, height) = (texture.width(), texture.height());
            for path in capture.screenshots.drain(..) {
                write_png(&path, &pixels, width, height, gpu.bgra())?;
            }
            if let Some(recorder) = &mut capture.recorder {
                let bgra = if gpu.bgra() { pixels } else { swap_red_blue(pixels) };
                recorder.frame(&bgra, width, height, frame, catalog)?;
            }
        }
        if let Some(audio) = self.audio.as_deref_mut() {
            audio.frame(frame, catalog);
        }
        if let Some(window) = self.window {
            let overlay = self.overlay.as_ref().is_some_and(|overlay| overlay.wants_pointer());
            window.set_cursor_visible(!frame.hide_cursor || overlay);
        }
        if let Some(preferences) = &frame.preferences {
            self.preferences.store(preferences)?;
        }
        Ok(())
    }

    fn report(&mut self, message: &str) {
        // The runtime's word on a new build starts "Blot reload".
        match message.strip_prefix("Blot reload") {
            Some(rest) if rest.starts_with("ed") => overlay::info(Source::Reload, message),
            Some(_) => overlay::error(Source::Reload, message),
            None => overlay::info(Source::Game, message),
        }
    }

    fn restored(&mut self) {
        self.preferences.replay(self.input);
        // The answers waiting were for the world that asked.
        self.services.forget();
    }
}

fn swap_red_blue(mut pixels: Vec<u8>) -> Vec<u8> {
    for texel in pixels.as_chunks_mut::<4>().0 {
        texel.swap(0, 2);
    }
    pixels
}

fn write_png(path: &Path, pixels: &[u8], width: u32, height: u32, bgra: bool) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut rgba = Vec::with_capacity(pixels.len());
    for texel in pixels.as_chunks::<4>().0 {
        let [a, b, c, _] = *texel;
        rgba.extend_from_slice(&if bgra { [c, b, a, 255] } else { [a, b, c, 255] });
    }
    image::save_buffer(path, &rgba, width, height, image::ColorType::Rgba8)
        .with_context(|| format!("writing {}", path.display()))?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

/// Writes the captured `texture` with the overlay drawn over it, as the
/// window shows it, from a copy: the capture itself stays the game's.
fn write_overlay_shot(gpu: &Gpu, texture: &wgpu::Texture, overlay: Option<&mut Box<dyn Overlay>>, path: &Path) -> Result<()> {
    let shot = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("overlay shot"),
        size: texture.size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: texture.format(),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_texture(texture.as_image_copy(), shot.as_image_copy(), texture.size());
    gpu.queue.submit([encoder.finish()]);
    match overlay {
        Some(overlay) => overlay.paint(&gpu.device, &gpu.queue, &shot),
        None => eprintln!("no overlay to draw into {}", path.display()),
    }
    let pixels = gpu.read(&shot)?;
    write_png(path, &pixels, shot.width(), shot.height(), gpu.bgra())
}

/// Draws `frame` offscreen until the renderer has loaded every asset and
/// faded in its detail and the veil has cleared, and no load, shader
/// compile or check is left whose event could reach the game later: from
/// then on its asset events cannot make two runs differ.
fn settle(gpu: &mut Gpu, frame: &RenderFrame, catalog: &Catalog) -> Result<()> {
    const PATIENCE: std::time::Duration = std::time::Duration::from_secs(60);
    let started = Instant::now();
    let mut draws = 0;
    while !gpu.renderer.quiet() || gpu.renderer.pending() > 0 {
        if started.elapsed() > PATIENCE {
            eprintln!("settle: assets still loading after {} s; starting anyway", PATIENCE.as_secs());
            return Ok(());
        }
        let texture = gpu.capture_texture();
        let output = Output {
            device: &gpu.device,
            queue: &gpu.queue,
            surface: None,
            size: gpu.size,
            blitter: None,
            overlay: None,
        };
        gpu.renderer.submit(frame, catalog, &mut CaptureTarget { output, texture })?;
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        if let Some(fault) = gpu.take_fault() {
            return Err(anyhow!(fault));
        }
        draws += 1;
        std::thread::sleep(std::time::Duration::from_millis(4));
    }
    eprintln!("settled in {:.0} ms ({draws} draws)", started.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

/// PEN_PROFILE_FRAMES: after PEN_PROFILE_WARMUP frames (90), sample this
/// many, report each phase's median and stop, as gdev's desktop host does.
pub struct Profile {
    warmup: usize,
    wanted: usize,
    timings: Vec<FrameTimings>,
    intervals: Vec<f64>,
    previous: Option<Instant>,
}

impl Profile {
    pub fn from_env() -> Result<Option<Profile>> {
        let Some(wanted) = std::env::var("PEN_PROFILE_FRAMES").ok() else { return Ok(None) };
        let warmup = std::env::var("PEN_PROFILE_WARMUP").ok().map(|value| value.parse()).transpose()?;
        Ok(Some(Profile {
            warmup: warmup.unwrap_or(90),
            wanted: wanted.parse()?,
            timings: Vec::new(),
            intervals: Vec::new(),
            previous: None,
        }))
    }

    /// Records a frame; true once enough have been sampled.
    fn sample(&mut self, frame: usize, timings: Option<FrameTimings>) -> bool {
        let now = Instant::now();
        if frame > self.warmup {
            if let Some(previous) = self.previous {
                self.intervals.push((now - previous).as_secs_f64() * 1000.0);
            }
            self.timings.extend(timings);
        }
        self.previous = Some(now);
        self.timings.len() >= self.wanted
    }

    fn report(&self, overlapped: bool) {
        let median = |mut values: Vec<f64>| {
            values.sort_by(f64::total_cmp);
            values.get(values.len() / 2).copied().unwrap_or(0.0)
        };
        let phase = |pick: fn(&FrameTimings) -> f64| median(self.timings.iter().map(pick).collect());
        eprintln!(
            "profile of {} {}frames (medians, ms): guest {:.2} (waited for {:.2}), decode {:.2}, validate {:.2}, \
             render+present {:.2}, housekeeping {:.2}, total {:.2}; frame interval {:.2}; packet {:.0} KB",
            self.timings.len(),
            if overlapped { "overlapped " } else { "" },
            phase(|t| t.guest_ms),
            phase(|t| t.wait_ms),
            phase(|t| t.decode_ms),
            phase(|t| t.validate_ms),
            phase(|t| t.submit_ms),
            phase(|t| t.housekeeping_ms),
            phase(|t| t.total_ms),
            median(self.intervals.clone()),
            phase(|t| t.packet_bytes as f64) / 1024.0,
        );
    }
}

pub struct Options {
    /// Stop after this many frames, or 0 to run until closed (or, headless,
    /// until the script ends).
    pub frames: usize,
    /// Write the last frame here.
    pub screenshot: Option<PathBuf>,
    /// Record every frame to this video.
    pub record: Option<PathBuf>,
    /// Steady game time: this many frames per game second.
    pub fps: Option<f64>,
    pub mute: bool,
    /// Reload the game and its assets when their files change.
    pub dev: bool,
    pub script: Vec<Step>,
    pub headless: bool,
    /// Before the game clock starts, draw the first frame until every asset
    /// has loaded and faded in, so later frames depend only on game time
    /// and input (repeatable shots, recordings without pop-in).
    pub settle: bool,
}

/// What a session does besides playing: replay a recording, record the
/// input, keep a history to travel in, overlap frames, trace packets.
#[derive(Clone, Debug, Default)]
pub struct Playback {
    /// Play this recording's input instead of the player's (`--replay`).
    pub replay: Option<PathBuf>,
    /// Record the input to this file (`--record-input`).
    pub record_input: Option<PathBuf>,
    /// Keep the last seconds to rewind and replay (`--history`; always
    /// while developing or when a script travels back).
    pub history: bool,
    /// Compute the next frame while this one draws (`--pipeline`).
    pub pipeline: bool,
    /// Write a line per frame: its position, and its input's and packet's
    /// hashes (`--trace`).
    pub trace: Option<PathBuf>,
}

/// Where the session stands in time, for the dev tools.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimelineStatus {
    /// Frames run so far: the next frame's position. What shows is the
    /// frame before it.
    pub frame: u64,
    /// The end of the recorded frames; beyond `frame` when paused after a
    /// rewind (stepping runs them again).
    pub latest: u64,
    /// The earliest frame `rewind_to` reaches; none without a history.
    pub earliest: Option<u64>,
    /// Game seconds of history before `frame`.
    pub seconds: f64,
    pub paused: bool,
    /// Recorded frames are playing (a replay).
    pub replaying: bool,
    /// A history is kept.
    pub history: bool,
    /// Frames overlap (`--pipeline`).
    pub pipelined: bool,
}

/// What a replay does when its frames run out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Replaying {
    /// `--replay`: stop (headless) or pause (a window).
    File,
    /// `replay_recent`: play on, live.
    Recent,
}

/// Seconds Ctrl+Shift+R replays.
pub const REPLAY_SECONDS: f64 = 5.0;

/// A frame on its way to the game: its input, and what to do with it.
struct Planned {
    position: u64,
    input: Vec<f32>,
    /// A recorded frame's loads, which it applies instead of reading files;
    /// None for a live frame.
    replayed: Option<Vec<Load>>,
    shots: Vec<PathBuf>,
    /// Shots with the overlay.
    overlay_shots: Vec<PathBuf>,
    saves: Vec<PathBuf>,
    loads: Vec<PathBuf>,
    /// The run ends after this frame.
    stop: bool,
}

pub struct Session {
    options: Options,
    paths: Paths,
    engine: wasmtime::Engine,
    pub gpu: Gpu,
    runtime: GameRuntime,
    audio: Option<Audio>,
    pub input: Input,
    preferences: Preferences,
    /// What the game asks of the platform and native libraries: called for
    /// live frames only, their answers the next live frame's input.
    pub services: Services,
    clock: Clock,
    script: Option<Script>,
    capture: Capture,
    dev: Option<Dev>,
    profile: Option<Profile>,
    /// Steps taken (frames run or paused ones presented).
    frames: usize,
    /// Paces a window's frames to game time (a steady clock, a replay):
    /// when it started and how much game time has passed since.
    pacer: Option<(Instant, f64)>,
    timeline: Timeline,
    paused: bool,
    /// Frames to run while paused.
    steps: u32,
    /// Positions to go back (or forward) to once the guest is idle.
    rewinds: VecDeque<Rewind>,
    replaying: Option<Replaying>,
    /// A new build waiting for the guest to be idle.
    build: Option<(GuestModule, Catalog)>,
    pipeline: bool,
    /// The frame the guest computes while the last one draws.
    flight: Option<Planned>,
    /// A script tick taken early that travels in time, so its frame waits.
    early: Option<Tick>,
    /// The frame last shown, shown again while paused.
    last: Option<RenderFrame>,
    input_recorder: Option<recording::Writer>,
    trace: Option<BufWriter<File>>,
    /// The renderer's asset events while recorded frames play: the game
    /// gets them once it plays live again.
    held: Vec<GameEvent>,
    /// Draws over the window's frames (`Overlay`).
    overlay: Option<Box<dyn Overlay>>,
    /// The last frame's timings, while measured.
    timings: Option<FrameTimings>,
    /// The input recording's file and the frame it starts at.
    recording_input: Option<(PathBuf, u64)>,
    /// Start (to a file) or stop recording input, once the guest is idle.
    record_request: Option<Option<PathBuf>>,
    /// A screenshot with the overlay of the last frame (`--overlay-shot`).
    final_overlay_shot: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
enum Rewind {
    To(u64),
    Replay(f64),
}

fn read_optional(path: &Path) -> Option<SavedState> {
    if !path.exists() {
        return None;
    }
    snapshot::read(path).map_err(|error| eprintln!("{error:#}")).ok()
}

/// Everything the game runtime may touch while a frame runs, borrowed from
/// the session's fields so the runtime and timeline stay free.
macro_rules! shell {
    ($session:expr, $window:expr) => {
        Shell {
            gpu: &mut $session.gpu,
            window: $window,
            audio: $session.audio.as_mut(),
            input: &mut $session.input,
            preferences: &mut $session.preferences,
            services: &mut $session.services,
            capture: &mut $session.capture,
            overlay: $session.overlay.as_mut(),
        }
    };
}

/// A save or load the frame or script asked for; a failure is reported, not fatal.
fn report(result: Result<()>, done: String) -> bool {
    match result {
        Ok(()) => {
            overlay::info(Source::Session, done);
            true
        }
        Err(error) => {
            overlay::error(Source::Session, format!("{error:#}"));
            false
        }
    }
}

/// Loads a quicksave or (with `scene`) the editor's scene from a file;
/// returns the state loaded, for the input recording.
fn load_file(runtime: &mut GameRuntime, host: &mut dyn Host, path: &Path, scene: bool) -> Option<Load> {
    let saved = snapshot::read(path).map_err(|error| eprintln!("{error:#}")).ok()?;
    let (result, what) = if scene {
        (runtime.load_scene(host, &saved), "scene ")
    } else {
        (runtime.load(host, &saved), "")
    };
    let load = if scene { Load::Scene(saved) } else { Load::Save(saved) };
    report(result, format!("loaded {what}{}", path.display())).then_some(load)
}

impl Session {
    pub fn start(
        options: Options,
        paths: Paths,
        engine: wasmtime::Engine,
        module: GuestModule,
        catalog: Catalog,
        mut gpu: Gpu,
        window: Option<&Window>,
    ) -> Result<Session> {
        gpu.renderer.preload(&catalog);
        // Headless runs and recordings play the sound into the recording, not the speakers.
        let audio = if options.mute || options.headless || options.record.is_some() {
            None
        } else {
            Audio::new(paths.assets.clone(), &catalog).map_err(|error| eprintln!("audio is off: {error:#}")).ok()
        };
        let fps = options.fps.or((options.headless || options.record.is_some()).then_some(60.0));
        let clock = match fps {
            Some(fps) => Clock::steady(fps),
            None => Clock::real(),
        };
        let recorder = match &options.record {
            Some(path) => Some(Recorder::start(
                path,
                gpu.size.0,
                gpu.size.1,
                fps.unwrap_or(60.0),
                (!options.mute).then_some((paths.assets.as_path(), &catalog)),
            )?),
            None => None,
        };
        let mut capture = Capture {
            recorder,
            offscreen: options.headless,
            screenshots: Vec::new(),
            overlay_shots: Vec::new(),
        };
        let mut input = Input::default();
        let mut preferences = Preferences::load(paths.preferences.clone());
        preferences.replay(&mut input);
        let mut services = Services::standard();
        let dev = if options.dev { Some(Dev::new(&paths.game, &paths.assets)?) } else { None };
        let mut audio = audio;
        let runtime = {
            let mut shell = Shell {
                gpu: &mut gpu,
                window,
                audio: audio.as_mut(),
                input: &mut input,
                preferences: &mut preferences,
                services: &mut services,
                capture: &mut capture,
                overlay: None,
            };
            GameRuntime::new(
                &engine,
                module,
                catalog,
                &mut shell,
                read_optional(&paths.quicksave),
                read_optional(paths.scene_to_read()),
            )?
        };
        let mut runtime = runtime;
        let first = runtime.take_drawn();
        if let Some(frame) = &first
            && options.settle
        {
            settle(&mut gpu, frame, runtime.catalog())?;
        }
        let profile = Profile::from_env()?;
        runtime.measure_frames(profile.is_some());
        let script = (!options.script.is_empty()).then(|| Script::new(options.script.clone()));
        let mut timeline = Timeline::default();
        if options.dev || script::travels_back(&options.script) {
            timeline.keep_history(Some(KEEP_SECONDS));
        }
        Ok(Session {
            options,
            paths,
            engine,
            gpu,
            runtime,
            audio,
            input,
            preferences,
            services,
            clock,
            script,
            capture,
            dev,
            profile,
            frames: 0,
            pacer: None,
            timeline,
            paused: false,
            steps: 0,
            rewinds: VecDeque::new(),
            replaying: None,
            build: None,
            pipeline: false,
            flight: None,
            early: None,
            last: first,
            input_recorder: None,
            trace: None,
            held: Vec::new(),
            overlay: None,
            timings: None,
            recording_input: None,
            record_request: None,
            final_overlay_shot: None,
        })
    }

    /// Sets up what the session does besides playing; before the first step.
    pub fn playback(&mut self, playback: Playback, window: Option<&Window>) -> Result<()> {
        self.pipeline = playback.pipeline;
        if playback.history || playback.replay.is_some() {
            self.timeline.keep_history(Some(KEEP_SECONDS));
        }
        if let Some(path) = &playback.trace {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = File::create(path).with_context(|| format!("writing {}", path.display()))?;
            self.trace = Some(BufWriter::new(file));
            self.runtime.hash_packets(true);
        }
        if let Some(path) = &playback.replay {
            let recording = recording::read(path)?;
            let start = recording.start.clone();
            let frames = recording.frames.len();
            let seconds: f64 = recording.frames.iter().map(Recorded::elapsed).sum();
            self.timeline.load(recording);
            let shell = shell!(self, window);
            let shown = self
                .runtime
                .restore(&mut Quiet(&shell), &start)
                .with_context(|| format!("replaying {}", path.display()))?;
            if self.options.settle {
                settle(&mut self.gpu, &shown, self.runtime.catalog())?;
            }
            self.last = Some(shown);
            self.services.forget();
            self.replaying = Some(Replaying::File);
            eprintln!("replaying {frames} frames ({seconds:.1} s of game time) from {}", path.display());
        }
        if let Some(path) = &playback.record_input {
            let moment = self.runtime.moment()?;
            self.input_recorder = Some(recording::Writer::create(path, self.timeline.position(), &moment)?);
            self.recording_input = Some((path.clone(), self.timeline.position()));
        }
        Ok(())
    }

    /// Stops the game where it is; the window keeps showing (and drawing)
    /// the last frame.
    pub fn pause(&mut self) {
        if !self.paused {
            self.paused = true;
            self.clock.pause();
            self.pacer = None;
        }
    }

    /// Plays on. After a rewind, play continues live from there, and the
    /// frames that had followed are forgotten; a paused replay plays on.
    pub fn resume(&mut self) {
        if !self.paused {
            return;
        }
        self.paused = false;
        self.steps = 0;
        if self.replaying.is_none() && self.timeline.next().is_some() {
            self.timeline.go_live();
        }
        self.clock.resume();
        self.pacer = None;
    }

    /// Runs one frame and stays paused (pausing first if need be). Paused
    /// after a rewind, it runs the next recorded frame again.
    pub fn step_frame(&mut self) {
        self.pause();
        self.steps += 1;
    }

    /// Goes to frame `frame` (pausing there) by restoring the nearest
    /// moment before it and running the recorded frames again unseen.
    /// Takes effect on the next step.
    pub fn rewind_to(&mut self, frame: u64) {
        self.pause();
        self.steps = 0;
        self.rewinds.push_back(Rewind::To(frame));
    }

    /// Plays the last `seconds` of recorded input again, on whatever build
    /// runs now (after a hot reload, the new one), then plays on live.
    /// Takes effect on the next step.
    pub fn replay_recent(&mut self, seconds: f64) {
        self.rewinds.push_back(Rewind::Replay(seconds));
    }

    pub fn timeline(&self) -> TimelineStatus {
        let frame = self.timeline.position();
        TimelineStatus {
            frame,
            latest: self.timeline.head().max(frame),
            earliest: self.timeline.earliest(),
            seconds: self.timeline.seconds(),
            paused: self.paused,
            replaying: self.replaying.is_some(),
            history: self.timeline.history(),
            pipelined: self.pipeline,
        }
    }

    #[cfg(feature = "devtools")]
    /// The frame `seconds` of game time before `frame`, as far back as the
    /// history reaches.
    pub fn frames_before(&self, frame: u64, seconds: f64) -> u64 {
        let earliest = self.timeline.earliest().unwrap_or(frame).min(frame);
        let (mut position, mut time) = (frame, 0.0);
        while position > earliest && time < seconds {
            position -= 1;
            time += self.timeline.recorded(position).map_or(STEP_SECONDS, Recorded::elapsed);
        }
        position
    }

    #[cfg(feature = "devtools")]
    /// Draws `overlay` over the window's frames from the next step on.
    pub fn set_overlay(&mut self, overlay: Box<dyn Overlay>) {
        self.overlay = Some(overlay);
    }

    /// A window event, for the overlay first: whether the game gets it too.
    pub fn overlay_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.overlay.as_mut().is_none_or(|overlay| overlay.window_event(window, event))
    }

    /// The interface the frame shown declares for assistive technology
    /// (`RenderFrame::ui`, see `accessibility`).
    pub fn interface(&self) -> &[UiNode] {
        self.last.as_ref().map_or(&[], |frame| &frame.ui)
    }

    /// The overlay's part of the accessibility tree: whether it shows, and
    /// its tree from its last layout, once.
    pub fn overlay_accesskit(&mut self) -> (bool, Option<accesskit::TreeUpdate>) {
        match &mut self.overlay {
            Some(overlay) => (overlay.accesskit_shown(), overlay.accesskit_update()),
            None => (false, None),
        }
    }

    /// An assistive technology's request for one of the overlay's nodes.
    pub fn overlay_accesskit_action(&mut self, request: accesskit::ActionRequest) {
        if let Some(overlay) = &mut self.overlay {
            overlay.accesskit_action(request);
        }
    }

    #[cfg(feature = "devtools")]
    /// Times each frame's phases (`frame_timings`), as PEN_PROFILE_FRAMES does.
    pub fn measure_frames(&mut self, enabled: bool) {
        self.runtime.measure_frames(enabled || self.profile.is_some());
    }

    #[cfg(feature = "devtools")]
    /// The last frame's timings, while measured.
    pub fn frame_timings(&self) -> Option<FrameTimings> {
        self.timings
    }

    #[cfg(feature = "devtools")]
    /// The game's linear memory, in bytes.
    pub fn guest_memory(&self) -> usize {
        self.runtime.memory_bytes()
    }

    #[cfg(feature = "devtools")]
    /// The running game's state, as a quicksave holds it (the overlay's
    /// Tunables panel reads its tuned values); none while an overlapped
    /// frame computes, or if the game cannot say.
    pub fn snapshot(&mut self) -> Option<SavedState> {
        if self.flight.is_some() {
            return None;
        }
        self.runtime.snapshot().map_err(|error| overlay::error(Source::Session, format!("{error:#}"))).ok()
    }

    #[cfg(feature = "devtools")]
    /// Keeps the last seconds to travel in from now on, as `--history` does
    /// from the start.
    pub fn keep_history(&mut self) {
        if !self.timeline.history() {
            self.timeline.keep_history(Some(KEEP_SECONDS));
        }
    }

    #[cfg(feature = "devtools")]
    /// Starts recording the input to `path` (a `.penrec`, from the frame
    /// it then is), or with None stops, once the guest is idle.
    pub fn record_input(&mut self, path: Option<PathBuf>) {
        self.record_request = Some(path);
    }

    #[cfg(feature = "devtools")]
    /// The input recording's file and the frame it starts at, while recording.
    pub fn input_recording(&self) -> Option<(&Path, u64)> {
        self.recording_input.as_ref().map(|(path, from)| (path.as_path(), *from))
    }

    #[cfg(feature = "devtools")]
    /// Writes the next frame shown with the overlay over it, as the window
    /// shows it (screenshots and recordings never have it).
    pub fn overlay_shot(&mut self, path: PathBuf) {
        self.capture.overlay_shots.push(path);
    }

    /// The same of the last frame (`--frames`), as `--screenshot` writes it.
    pub fn overlay_shot_at_end(&mut self, path: PathBuf) {
        self.final_overlay_shot = Some(path);
    }

    fn stop_recording(&mut self) -> Result<()> {
        if let Some(recorder) = self.input_recorder.take() {
            recorder.finish()?;
            if let Some((path, from)) = self.recording_input.take() {
                let frames = self.timeline.position().saturating_sub(from);
                overlay::note(Source::Session, format!("recorded {frames} frames of input to {}", path.display()));
            }
        }
        Ok(())
    }

    /// Records the input to `path` from here, once the guest is idle. A
    /// replay starts by restoring the recording's moment, so the game is
    /// restored from it here too (as a rewind does, the clock running on):
    /// what it keeps outside its state then starts over in both alike, and
    /// the replay gives the packets this run does.
    fn start_recording(&mut self, window: Option<&Window>, path: PathBuf) -> Result<()> {
        self.stop_recording()?;
        let moment = self.runtime.moment()?;
        let position = self.timeline.position();
        self.input_recorder = Some(recording::Writer::create(&path, position, &moment)?);
        let shell = shell!(self, window);
        let shown = if self.timeline.history() {
            // Rewinds after it then restore from no earlier than its start.
            self.timeline.keep_moment(moment);
            self.timeline.restart(&mut self.runtime, &shell, position)?.0
        } else {
            self.runtime.restore(&mut Quiet(&shell), &moment)?
        };
        self.runtime.continue_clock(self.clock.now());
        self.last = Some(shown);
        overlay::info(Source::Session, format!("recording input to {} from frame {position}", path.display()));
        self.recording_input = Some((path, position));
        Ok(())
    }

    /// Going back to `at`: the input recording goes back too, or ends if
    /// it started later.
    fn rewind_recording(&mut self, at: u64) -> Result<()> {
        if let Some((path, from)) = &self.recording_input
            && at < *from
        {
            overlay::info(Source::Session, format!("stopped recording {}: went back before it started", path.display()));
            self.stop_recording()?;
        }
        if let Some(recorder) = &mut self.input_recorder {
            recorder.rewind(at)?;
        }
        Ok(())
    }

    /// The frame a rewind queued now starts from.
    fn target(&self) -> u64 {
        match self.rewinds.back() {
            Some(Rewind::To(frame)) => *frame,
            _ => self.timeline.position(),
        }
    }

    fn hotkey(&mut self, hotkey: Hotkey) {
        match hotkey {
            Hotkey::Pause if self.paused => {
                self.resume();
                overlay::info(Source::Session, "playing");
            }
            Hotkey::Pause => {
                self.pause();
                let status = self.timeline();
                let back = if status.history { ", Ctrl+Shift+comma back" } else { "" };
                overlay::info(
                    Source::Session,
                    format!("paused at frame {} (Ctrl+Shift+period steps{back}, Ctrl+Shift+P plays)", status.frame),
                );
            }
            Hotkey::Step => self.step_frame(),
            Hotkey::Back => self.rewind_to(self.target().saturating_sub(1)),
            Hotkey::Replay => self.replay_recent(REPLAY_SECONDS),
        }
    }

    fn travel(&mut self, travel: Travel) {
        match travel {
            Travel::Pause => self.pause(),
            Travel::Resume => self.resume(),
            Travel::Step(frames) => {
                self.pause();
                self.steps += frames;
            }
            Travel::Rewind(frames) => self.rewind_to(self.target().saturating_sub(frames)),
            Travel::Replay(seconds) => self.replay_recent(seconds),
        }
    }

    /// Whether the game plays the player's input (not recorded frames).
    fn live(&self) -> bool {
        self.replaying.is_none() && self.timeline.next().is_none()
    }

    /// Goes to `target`, paused.
    fn rewind(&mut self, window: Option<&Window>, target: u64) -> Result<()> {
        let started = Instant::now();
        let shell = shell!(self, window);
        let trace = &mut self.trace;
        // The trace shows the frames run again unseen too: a replay of the
        // input recording runs them as they ran last.
        let mut unseen = |position: u64, recorded: &Recorded, runtime: &GameRuntime| -> Result<()> {
            if let Some(trace) = trace {
                writeln!(trace, "{position} {:016x} {:016x}", hash_lanes(&recorded.input), runtime.packet_hash())?;
            }
            Ok(())
        };
        let (shown, at) = self.timeline.rewind(&mut self.runtime, &shell, target, &mut unseen)?;
        self.services.forget();
        let position = self.timeline.position();
        self.runtime.continue_clock(self.clock.now());
        self.last = Some(shown);
        self.pacer = None;
        // The recording goes back to the moment restored, and on again
        // from there marked as restored.
        self.rewind_recording(at)?;
        if let Some(recorder) = &mut self.input_recorder {
            for frame in at..position {
                recorder.frame(self.timeline.recorded(frame).context("a rewound frame is gone")?)?;
            }
        }
        let ahead = self.timeline.head() - position;
        overlay::info(
            Source::Session,
            format!(
                "at frame {position} in {:.0} ms ({ahead} recorded frames ahead: Ctrl+Shift+period runs them, \
                 Ctrl+Shift+P plays on from here)",
                started.elapsed().as_secs_f64() * 1000.0
            ),
        );
        Ok(())
    }

    /// Plays the last `seconds` of recorded frames again.
    fn replay(&mut self, window: Option<&Window>, seconds: f64) -> Result<()> {
        let from = self.timeline.seconds_back(seconds);
        let shell = shell!(self, window);
        let (shown, at) = self.timeline.restart(&mut self.runtime, &shell, from)?;
        self.services.forget();
        self.last = Some(shown);
        self.replaying.get_or_insert(Replaying::Recent);
        self.paused = false;
        self.steps = 0;
        self.clock.resume();
        self.pacer = None;
        self.rewind_recording(at)?;
        let frames = self.timeline.head() - at;
        overlay::info(Source::Session, format!("replaying {frames} frames from frame {at} on the running build"));
        Ok(())
    }

    /// A replay's frames ran out; returns whether the run ends.
    fn replay_ended(&mut self) -> bool {
        let replaying = self.replaying.take();
        self.runtime.continue_clock(self.clock.now());
        let position = self.timeline.position();
        match replaying {
            Some(Replaying::File) if self.options.frames == 0 && self.options.headless => {
                overlay::info(Source::Session, format!("the replay ended at frame {position}"));
                true
            }
            Some(Replaying::File) if self.options.frames == 0 => {
                self.pause();
                overlay::info(
                    Source::Session,
                    format!("the replay ended at frame {position}; paused (Ctrl+Shift+P plays on, live)"),
                );
                false
            }
            Some(_) => {
                overlay::info(Source::Session, format!("the replay ended at frame {position}; playing live"));
                false
            }
            None => false,
        }
    }

    /// One frame of the script: its input goes to the game, the rest is
    /// done around the frame.
    fn tick(&mut self) -> Tick {
        let viewport = Viewport { width: self.gpu.size.0, height: self.gpu.size.1 };
        let mut tick = self.script.as_mut().map(|script| script.tick(viewport)).unwrap_or_default();
        for event in tick.events.drain(..) {
            self.input.enqueue(event);
        }
        if let Some(overlay) = &mut self.overlay {
            for line in tick.overlay.drain(..) {
                overlay.command(line);
            }
        }
        tick
    }

    /// The script ran out (headless, without --frames): stop.
    fn script_over(&self) -> bool {
        self.script.as_ref().is_some_and(Script::done) && self.options.headless && self.options.frames == 0
    }

    /// Waits until a frame's game time has come (a window plays at the
    /// game's speed).
    fn pace(&mut self, elapsed: f64) {
        let (started, time) = self.pacer.get_or_insert_with(|| (Instant::now(), 0.0));
        *time += elapsed;
        let due = *started + std::time::Duration::from_secs_f64(*time);
        if let Some(wait) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
    }

    /// The next frame's input (recorded, or the player's) and its script
    /// tick's requests. Keeps a moment first when the history wants one,
    /// and restores where a replayed run was restored.
    fn plan(&mut self, window: Option<&Window>, tick: Tick) -> Result<Planned> {
        let shell = shell!(self, window);
        self.timeline.prepare(&mut self.runtime, &shell)?;
        self.frames += 1;
        let last = self.options.frames != 0 && self.frames >= self.options.frames;
        let viewport = Viewport { width: self.gpu.size.0, height: self.gpu.size.1 };
        let (input, replayed) = match self.timeline.next() {
            Some(recorded) => {
                // Recorded frames bring their own input.
                let planned = (recorded.input.clone(), Some(recorded.loads.clone()));
                self.input.discard();
                planned
            }
            None => {
                let frame = self.input.next_frame(self.clock.next(), viewport);
                (self.runtime.input(&frame)?, None)
            }
        };
        if window.is_some() && (self.clock.is_steady() || replayed.is_some()) {
            self.pace(f64::from(input[0]));
        }
        let mut shots: Vec<PathBuf> = tick.shots.iter().map(|name| self.paths.shots.join(format!("{name}.png"))).collect();
        if last && let Some(path) = &self.options.screenshot {
            shots.push(path.clone());
        }
        let overlay_shots = self.final_overlay_shot.iter().filter(|_| last).cloned().collect();
        Ok(Planned {
            position: self.timeline.position(),
            input,
            replayed,
            shots,
            overlay_shots,
            saves: tick.saves,
            loads: tick.loads,
            stop: last || tick.quit || self.script_over(),
        })
    }

    /// Whether the next frame can go to the guest before this one draws:
    /// nothing this frame does afterwards needs the guest or changes its
    /// state, and the next frame is the player's.
    fn overlaps(&self, planned: &Planned, rendered: &RenderFrame) -> bool {
        self.pipeline
            && !planned.stop
            && planned.replayed.is_none()
            && planned.saves.is_empty()
            && planned.loads.is_empty()
            && !self.paused
            && self.live()
            && self.build.is_none()
            && self.rewinds.is_empty()
            // Its requests' answers go to the next frame, as without overlap.
            && rendered.services.is_empty()
            && !(rendered.checkpoint_requested
                || rendered.undo_requested
                || rendered.redo_requested
                || rendered.save_requested
                || rendered.load_requested
                || rendered.scene_save_requested
                || rendered.scene_load_requested
                || rendered.exit_requested)
    }

    /// A frame ran: into the history, the input recording and the trace.
    fn ran(&mut self, position: u64, input: Vec<f32>, loads: Vec<Load>) -> Result<()> {
        debug_assert_eq!(position, self.timeline.position());
        if let Some(trace) = &mut self.trace {
            writeln!(trace, "{position} {:016x} {:016x}", hash_lanes(&input), self.runtime.packet_hash())?;
        }
        let recorded = Recorded { input, loads, restored: self.timeline.restored_here() };
        if let Some(recorder) = &mut self.input_recorder {
            recorder.frame(&recorded)?;
        }
        self.timeline.ran(recorded);
        Ok(())
    }

    fn reload(&mut self, window: Option<&Window>, module: GuestModule, catalog: Catalog) {
        let mut shell = shell!(self, window);
        if self.runtime.reload(&mut shell, module, catalog) == ReloadResult::Activated {
            self.timeline.rebuilt();
            if let Some(drawn) = self.runtime.take_drawn() {
                self.last = Some(drawn);
            }
            if self.timeline.history() {
                overlay::info(Source::Reload, format!("dev: Ctrl+Shift+R replays the last {REPLAY_SECONDS} s on the new build"));
            }
        }
    }

    /// Takes reloads as they come: assets at once, a new build when the
    /// guest is idle.
    fn poll_dev(&mut self) {
        let updates = match &mut self.dev {
            Some(dev) => dev.poll(&self.engine),
            None => Vec::new(),
        };
        for update in updates {
            match update {
                Update::Build(module, catalog) => {
                    self.gpu.renderer.preload(&catalog);
                    if let Some(audio) = &mut self.audio {
                        audio.preload(&catalog);
                    }
                    self.build = Some((module, catalog));
                }
                Update::Assets(keys) => {
                    self.gpu.renderer.reload(&keys);
                    if let Some(audio) = &mut self.audio {
                        audio.reload(&keys);
                    }
                }
            }
        }
    }

    /// Paused: the last frame again (without its sounds, and not into a
    /// recording), and the script's shots of it.
    fn present_paused(&mut self, window: Option<&Window>, tick: Tick) -> Result<bool> {
        self.frames += 1;
        let last = self.options.frames != 0 && self.frames >= self.options.frames;
        for name in &tick.shots {
            self.capture.screenshots.push(self.paths.shots.join(format!("{name}.png")));
        }
        if last && let Some(path) = &self.options.screenshot {
            self.capture.screenshots.push(path.clone());
        }
        if last && let Some(path) = &self.final_overlay_shot {
            self.capture.overlay_shots.push(path.clone());
        }
        if let Some(frame) = self.last.take() {
            let recorder = self.capture.recorder.take();
            let mut shell = shell!(self, window);
            shell.audio = None;
            let result = shell.render(&frame, self.runtime.catalog());
            self.capture.recorder = recorder;
            self.last = Some(frame);
            result?;
            if let Some(fault) = self.gpu.take_fault() {
                return Err(anyhow!(fault));
            }
        }
        Ok(!(last || tick.quit || self.script_over()))
    }

    /// Runs one frame (or, paused, shows the last again); returns whether
    /// to keep going.
    pub fn step(&mut self, window: Option<&Window>) -> Result<bool> {
        if self.gpu.size.0 == 0 || self.gpu.size.1 == 0 {
            return Ok(true);
        }
        // Laid out before the frame's time starts: its drawing counts in render+present.
        if let Some(mut overlay) = self.overlay.take() {
            overlay.update(self, window);
            self.overlay = Some(overlay);
        }
        let started = Instant::now();
        self.poll_dev();
        for event in self.gpu.renderer.take_events() {
            // While developing, say when a reloaded asset is in (or why not).
            if self.dev.is_some() {
                match &event {
                    GameEvent::AssetReady { key, revision, .. } if *revision > 1 => {
                        overlay::info(Source::Asset, format!("dev: {key} is in (revision {revision})"));
                    }
                    GameEvent::AssetFailed { key, message, .. } => {
                        overlay::error(Source::Asset, format!("dev: {key} failed: {message}"));
                    }
                    _ => {}
                }
            }
            self.held.push(event);
        }
        self.services.frame(&mut self.input);
        if self.live() {
            for event in self.held.drain(..) {
                self.input.enqueue(event);
            }
            for event in self.services.take_answers() {
                self.input.enqueue(event);
            }
        }
        for hotkey in self.input.take_hotkeys() {
            self.hotkey(hotkey);
        }
        let planned = match self.flight.take() {
            Some(planned) => planned,
            None => {
                // The guest is idle: what waited for it happens now.
                if let Some((module, catalog)) = self.build.take() {
                    self.reload(window, module, catalog);
                }
                let tick = match self.early.take() {
                    Some(tick) => tick,
                    None => self.tick(),
                };
                if let Some(travel) = tick.travel {
                    self.travel(travel);
                }
                while let Some(rewind) = self.rewinds.pop_front() {
                    // Of rewinds queued one after another (a held key), the last.
                    if matches!((rewind, self.rewinds.front()), (Rewind::To(_), Some(Rewind::To(_)))) {
                        continue;
                    }
                    let result = match rewind {
                        Rewind::To(frame) => self.rewind(window, frame),
                        Rewind::Replay(seconds) => self.replay(window, seconds),
                    };
                    if let Err(error) = result {
                        overlay::error(Source::Session, format!("time travel: {error:#}"));
                    }
                }
                if let Some(path) = self.record_request.take()
                    && let Err(error) = match path {
                        Some(path) => self.start_recording(window, path),
                        None => self.stop_recording(),
                    }
                {
                    overlay::error(Source::Session, format!("recording input: {error:#}"));
                }
                if self.paused && self.steps == 0 {
                    return self.present_paused(window, tick);
                }
                self.steps = self.steps.saturating_sub(1);
                let planned = self.plan(window, tick)?;
                self.runtime.send(&planned.input)?;
                planned
            }
        };
        let rendered = {
            let shell = shell!(self, window);
            self.runtime.receive(&shell)?
        };
        let overlapped = self.overlaps(&planned, &rendered);
        let mut planned = planned;
        if overlapped {
            // This frame changes nothing after it draws: the next one goes
            // to the guest now and computes while this one draws.
            self.ran(planned.position, std::mem::take(&mut planned.input), Vec::new())?;
            let tick = self.tick();
            if tick.travel.is_some() {
                self.early = Some(tick);
            } else {
                let next = self.plan(window, tick)?;
                self.runtime.send(&next.input)?;
                self.flight = Some(next);
            }
        }
        let stop = self.complete(window, planned, rendered, overlapped, started)?;
        Ok(!stop)
    }

    /// Draws a frame and does what it and its script tick asked; returns
    /// whether the run ends.
    fn complete(
        &mut self,
        window: Option<&Window>,
        planned: Planned,
        rendered: RenderFrame,
        overlapped: bool,
        started: Instant,
    ) -> Result<bool> {
        self.capture.screenshots.extend(planned.shots);
        self.capture.overlay_shots.extend(planned.overlay_shots);
        let submitting = Instant::now();
        let mut loads = Vec::new();
        let submit_ms;
        {
            let mut shell = shell!(self, window);
            shell.render(&rendered, self.runtime.catalog())?;
            submit_ms = submitting.elapsed().as_secs_f64() * 1000.0;
            if let Some(fault) = shell.gpu.take_fault() {
                return Err(anyhow!(fault));
            }
            // A live frame's requests (a recorded one's answers are in the
            // recording), before an undo or load replaces the world asking.
            if planned.replayed.is_none() {
                shell.services.answer(&rendered.services);
            }
            self.runtime.follow(&mut shell, &rendered)?;
            match &planned.replayed {
                // The states it loaded, not the files: they may have changed
                // since. A replay writes no files.
                Some(recorded) => {
                    for load in recorded {
                        timeline::apply(&mut self.runtime, &mut shell, load)?;
                    }
                    loads.clone_from(recorded);
                }
                None => {
                    let paths = &self.paths;
                    if rendered.scene_save_requested {
                        report(self.runtime.save_scene(&paths.scene), format!("saved scene {}", paths.scene.display()));
                    }
                    if rendered.scene_load_requested {
                        loads.extend(load_file(&mut self.runtime, &mut shell, paths.scene_to_read(), true));
                    }
                    if rendered.save_requested {
                        report(self.runtime.save(&paths.quicksave), format!("saved {}", paths.quicksave.display()));
                    }
                    if rendered.load_requested {
                        loads.extend(load_file(&mut self.runtime, &mut shell, &paths.quicksave, false));
                    }
                    for path in &planned.saves {
                        report(self.runtime.save(path), format!("saved {}", path.display()));
                    }
                    for path in &planned.loads {
                        loads.extend(load_file(&mut self.runtime, &mut shell, path, false));
                    }
                }
            }
        }
        if !overlapped {
            self.ran(planned.position, planned.input, loads)?;
        }
        let exit = rendered.exit_requested;
        self.last = Some(self.runtime.take_drawn().unwrap_or(rendered));
        let mut stop = exit || planned.stop;
        if self.replaying.is_some() && self.timeline.next().is_none() {
            stop |= self.replay_ended();
        }
        // Only while measured (a profile, or the overlay's profiler).
        self.timings = self.runtime.last_timings().map(|mut timings| {
            timings.submit_ms = submit_ms;
            timings.total_ms = started.elapsed().as_secs_f64() * 1000.0;
            timings.housekeeping_ms =
                (timings.total_ms - timings.wait_ms - timings.decode_ms - timings.validate_ms - timings.submit_ms)
                    .max(0.0);
            timings
        });
        if let Some(profile) = &mut self.profile
            && profile.sample(self.frames, self.timings)
        {
            profile.report(self.pipeline);
            return Ok(true);
        }
        Ok(stop)
    }

    /// Stops the game and finishes any recording; the GPU, with its loaded
    /// assets, can start another session.
    pub fn finish(mut self) -> Result<Gpu> {
        self.runtime.close();
        self.services.report();
        if let Some(recorder) = self.input_recorder.take() {
            recorder.finish()?;
        }
        if let Some(mut trace) = self.trace.take() {
            trace.flush()?;
        }
        if let Some(recorder) = self.capture.recorder.take() {
            recorder.finish()?;
        }
        Ok(self.gpu)
    }
}
