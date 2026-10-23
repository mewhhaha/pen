//! The session's GPU: the device and the game's renderer, and what frames
//! are drawn into and shown on (the window's swapchain, or an offscreen
//! image to capture, show again while paused or read back for shots and
//! recordings).
use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use pen_protocol::{Catalog, RenderFrame};
use pen_render::RenderTarget;
use pen_runtime::game::DrawTimings;
use winit::window::Window;

use super::Overlay;

/// What the last frame drawn held, for the dev tools' diagnostics.
#[cfg(feature = "devtools")]
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
    pub draw_timings: DrawTimings,
    #[cfg(feature = "devtools")]
    pub drawn: Drawn,
    /// The window's swapchain; none when headless.
    pub surface: Option<(wgpu::Surface<'static>, wgpu::SurfaceConfiguration)>,
    /// Shows frames drawn offscreen in the window, scaled to fit.
    pub(super) blitter: Option<wgpu::util::TextureBlitter>,
    /// A frame size the window does not change (a recording's).
    pinned: Option<(u32, u32)>,
    /// The offscreen image frames are captured from.
    capture: Option<wgpu::Texture>,
    /// The paused frame's image, shown again without drawing it again.
    pub(super) still: Option<wgpu::Texture>,
    fault: Fault,
}

/// An uncaptured GPU error or a lost device, which ends the run.
type Fault = Arc<Mutex<Option<String>>>;

/// The adapter to draw with: the fast one, or with `software` (`--software`
/// or PEN_SOFTWARE=1) a software one, such as Mesa's lavapipe on Vulkan or
/// WARP on DX12, as on CI machines without a GPU.
fn request_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
    software: bool,
) -> Result<wgpu::Adapter> {
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: software,
        compatible_surface: surface,
        ..Default::default()
    }));
    crate::startup::mark("GPU adapter found");
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
    let descriptor =
        wgpu::DeviceDescriptor { required_features: pen_render::optional_features(adapter), ..Default::default() };
    let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))?;
    crate::startup::mark("GPU device open");
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

fn renderer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    display: pen_render::Display,
    assets: &Path,
) -> Result<pen_render::Renderer> {
    pen_render::Renderer::new(
        device.clone(),
        queue.clone(),
        pen_render::RendererOptions {
            format,
            display,
            assets: assets.to_owned(),
            veil: Some(0.6),
            // Compiled pipelines, kept per adapter between runs.
            pipeline_cache: dirs::cache_dir().map(|cache| cache.join("pen/pipelines")),
        },
    )
}

impl Gpu {
    /// With `pinned`, frames keep that size whatever the window's;
    /// `software` draws on a software adapter; `hdr` shows HDR on a display
    /// and surface that can.
    pub fn windowed(
        display: Box<dyn wgpu::wgt::WgpuHasDisplayHandle>,
        window: Arc<Window>,
        present_mode: wgpu::PresentMode,
        assets: &Path,
        pinned: Option<(u32, u32)>,
        software: bool,
        hdr: bool,
    ) -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(display));
        let surface = instance.create_surface(window.clone())?;
        let adapter = request_adapter(&instance, Some(&surface), software)?;
        let (device, queue, fault) = open_device(&adapter)?;
        let capabilities = surface.get_capabilities(&adapter);
        // Like the browser's preferred canvas format: 8 bits and not sRGB,
        // since the present pass encodes colours itself; or an HDR one.
        let (format, shown) = pen_render::choose_surface(&capabilities, &surface.display_hdr_info(&adapter), hdr);
        if shown.is_hdr() {
            eprintln!("HDR: {:?} on {format:?}, {:.1}x SDR white", shown.color_space, shown.headroom);
        }
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
            color_space: shown.color_space,
        };
        surface.configure(&device, &config);
        let renderer = renderer(&device, &queue, format, shown, assets)?;
        crate::startup::mark("GPU ready");
        let blitter =
            wgpu::util::TextureBlitterBuilder::new(&device, format).sample_type(wgpu::FilterMode::Linear).build();
        Ok(Gpu {
            size: pinned.unwrap_or((config.width, config.height)),
            draw_timings: DrawTimings::default(),
            surface: Some((surface, config)),
            blitter: Some(blitter),
            pinned,
            device,
            queue,
            renderer,
            format,
            capture: None,
            still: None,
            #[cfg(feature = "devtools")]
            drawn: Drawn::default(),
            fault,
        })
    }

    /// No window: every frame draws offscreen at `size`.
    pub fn headless(size: (u32, u32), assets: &Path, software: bool) -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = request_adapter(&instance, None, software)?;
        let (device, queue, fault) = open_device(&adapter)?;
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let renderer = renderer(&device, &queue, format, pen_render::Display::SDR, assets)?;
        crate::startup::mark("GPU ready");
        Ok(Gpu {
            device,
            queue,
            renderer,
            format,
            size,
            draw_timings: DrawTimings::default(),
            surface: None,
            blitter: None,
            pinned: Some(size),
            capture: None,
            still: None,
            #[cfg(feature = "devtools")]
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

    pub(super) fn take_fault(&self) -> Option<String> {
        self.fault.lock().unwrap().take()
    }

    pub(super) fn capture_texture(&mut self) -> wgpu::Texture {
        offscreen(&self.device, &mut self.capture, self.size, self.format, "capture")
    }

    pub(super) fn still_texture(&mut self) -> wgpu::Texture {
        offscreen(&self.device, &mut self.still, self.size, self.format, "still")
    }

    /// The texture's pixels as BGRA (or RGBA for an RGBA format), rows
    /// packed, waited for.
    pub(super) fn read(&self, texture: &wgpu::Texture) -> Result<Vec<u8>> {
        let (buffer, row, _) = copy_out(&self.device, &self.queue, texture, None);
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        let mapped = slice.get_mapped_range().map_err(|error| anyhow!("reading a frame back: {error:?}"))?;
        Ok(packed(&mapped, texture.width(), texture.height(), row))
    }

    pub(super) fn bgra(&self) -> bool {
        matches!(self.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb)
    }
}

/// Copies `texture` into `buffer` (or a new one) for reading back, its rows
/// padded to the copy alignment; returns the buffer, its row pitch and the
/// copy's submission.
fn copy_out(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    buffer: Option<wgpu::Buffer>,
) -> (wgpu::Buffer, u32, wgpu::SubmissionIndex) {
    let (width, height) = (texture.width(), texture.height());
    let row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let size = u64::from(row * height);
    let buffer = buffer.filter(|buffer| buffer.size() == size).unwrap_or_else(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(height) },
        },
        texture.size(),
    );
    let submitted = queue.submit([encoder.finish()]);
    (buffer, row, submitted)
}

/// A mapped readback's pixels with their rows packed: one copy when the
/// rows have no padding (a width of a multiple of 64), else row by row.
fn packed(mapped: &[u8], width: u32, height: u32, row: u32) -> Vec<u8> {
    let (packed, height) = (width as usize * 4, height as usize);
    if row as usize == packed {
        return mapped[..packed * height].to_vec();
    }
    let mut pixels = Vec::with_capacity(packed * height);
    for line in mapped.chunks(row as usize).take(height) {
        pixels.extend_from_slice(&line[..packed]);
    }
    pixels
}

/// Frames read back at most at once for a recording.
const READBACK_RING: usize = 3;

/// A recording's frames on their way back from the GPU: each is copied into
/// a buffer that maps without waiting and read once it has, in order, while
/// the frames after it draw. The frame loop waits only when
/// `READBACK_RING` are under way.
#[derive(Default)]
pub(super) struct Readback {
    flight: VecDeque<Reading>,
    /// Buffers read and unmapped, for the next frames.
    spare: Vec<wgpu::Buffer>,
}

struct Reading {
    buffer: wgpu::Buffer,
    size: (u32, u32),
    row: u32,
    mapped: Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
}

/// A frame read back: its pixels, rows packed, and its size.
pub(super) type Pixels = (Vec<u8>, (u32, u32));

impl Readback {
    /// Starts reading `texture` back; returns the frames that came back
    /// since, oldest first.
    pub(super) fn start(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<Vec<Pixels>> {
        let (buffer, row, _) = copy_out(device, queue, texture, self.spare.pop());
        let (sender, mapped) = channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.flight.push_back(Reading { buffer, size: (texture.width(), texture.height()), row, mapped });
        self.take(device, self.flight.len() > READBACK_RING)
    }

    /// The frames that came back, oldest first; with `wait`, at least the
    /// oldest under way, waited for.
    fn take(&mut self, device: &wgpu::Device, mut wait: bool) -> Result<Vec<Pixels>> {
        device.poll(wgpu::PollType::Poll)?;
        let mut taken = Vec::new();
        while let Some(reading) = self.flight.front() {
            let mapped = match reading.mapped.try_recv() {
                Ok(mapped) => mapped,
                Err(TryRecvError::Empty) if wait => {
                    device.poll(wgpu::PollType::wait_indefinitely())?;
                    continue;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => Err(wgpu::BufferAsyncError),
            };
            wait = false;
            let reading = self.flight.pop_front().expect("the frame just looked at");
            mapped.map_err(|error| anyhow!("reading a frame back: {error}"))?;
            {
                let view = reading
                    .buffer
                    .slice(..)
                    .get_mapped_range()
                    .map_err(|error| anyhow!("reading a frame back: {error:?}"))?;
                taken.push((packed(&view, reading.size.0, reading.size.1, reading.row), reading.size));
            }
            reading.buffer.unmap();
            self.spare.push(reading.buffer);
        }
        Ok(taken)
    }

    /// Every frame still under way, waited for.
    pub(super) fn finish(&mut self, device: &wgpu::Device) -> Result<Vec<Pixels>> {
        let mut taken = Vec::new();
        while !self.flight.is_empty() {
            taken.extend(self.take(device, true)?);
        }
        Ok(taken)
    }
}

/// The texture in `slot` if it has `size`, else a new one there: an image
/// to draw frames into, copy to the window and read back.
fn offscreen(
    device: &wgpu::Device,
    slot: &mut Option<wgpu::Texture>,
    (width, height): (u32, u32),
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::Texture {
    if let Some(texture) = slot.as_ref().filter(|texture| (texture.width(), texture.height()) == (width, height)) {
        return texture.clone();
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    *slot = Some(texture.clone());
    texture
}

/// What a frame is presented to: the GPU fields a target needs, borrowed
/// apart from the renderer that draws into it.
pub(super) struct Output<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) queue: &'a wgpu::Queue,
    pub(super) surface: Option<&'a (wgpu::Surface<'static>, wgpu::SurfaceConfiguration)>,
    pub(super) size: (u32, u32),
    pub(super) blitter: Option<&'a wgpu::util::TextureBlitter>,
    /// Draws over the swapchain image before it is presented.
    pub(super) overlay: Option<&'a mut Box<dyn Overlay>>,
    pub(super) timings: &'a mut DrawTimings,
}

impl Output<'_> {
    fn acquire(&mut self) -> Option<wgpu::SurfaceTexture> {
        let (surface, config) = self.surface?;
        let started = Instant::now();
        let acquired = surface.get_current_texture();
        self.timings.acquire_ms += started.elapsed().as_secs_f64() * 1000.0;
        match acquired {
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

    fn present(&mut self, texture: wgpu::SurfaceTexture) {
        let started = Instant::now();
        self.queue.present(texture);
        self.timings.present_ms += started.elapsed().as_secs_f64() * 1000.0;
    }
}

/// The window's swapchain image for one frame.
pub(super) struct SurfaceTarget<'a> {
    pub(super) output: Output<'a>,
    pub(super) texture: Option<wgpu::SurfaceTexture>,
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
            self.output.present(texture);
        }
    }
}

/// An offscreen image to capture; a windowed run also copies it to the
/// swapchain so the window shows what is captured.
pub(super) struct CaptureTarget<'a> {
    pub(super) output: Output<'a>,
    pub(super) texture: wgpu::Texture,
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
        self.output.present(swapchain);
    }
}

pub(super) fn write_png(path: &Path, pixels: &[u8], width: u32, height: u32, bgra: bool) -> Result<()> {
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
pub(super) fn write_overlay_shot(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    overlay: Option<&mut Box<dyn Overlay>>,
    path: &Path,
) -> Result<()> {
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
pub(super) fn settle(gpu: &mut Gpu, frame: &RenderFrame, catalog: &Catalog) -> Result<()> {
    const PATIENCE: std::time::Duration = std::time::Duration::from_secs(60);
    // A software adapter (lavapipe, WARP) compiles each pipeline on the CPU,
    // seconds apiece, on CI machines of two or four cores.
    let software = gpu.device.adapter_info().device_type == wgpu::DeviceType::Cpu;
    let patience = if software { PATIENCE * 10 } else { PATIENCE };
    let started = Instant::now();
    let mut draws = 0;
    while !gpu.renderer.quiet() || gpu.renderer.pending() > 0 {
        if started.elapsed() > patience {
            eprintln!("settle: assets still loading after {} s; starting anyway", patience.as_secs());
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
            timings: &mut DrawTimings::default(),
        };
        gpu.renderer.submit(frame, catalog, &mut CaptureTarget { output, texture })?;
        crate::startup::drawn(&gpu.renderer);
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        if let Some(fault) = gpu.take_fault() {
            return Err(anyhow!(fault));
        }
        draws += 1;
        // Until the loader brings something, or about a display refresh
        // passes while detail fades in.
        gpu.renderer.wait(std::time::Duration::from_millis(16));
    }
    eprintln!("settled in {:.0} ms ({draws} draws)", started.elapsed().as_secs_f64() * 1000.0);
    crate::startup::mark("settled");
    Ok(())
}
