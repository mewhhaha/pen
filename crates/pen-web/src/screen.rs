//! The canvas on WebGPU, or on WebGL2 where the browser has no WebGPU: the
//! game's renderer, and the dev tools' sharing its device, drawing into the
//! canvas's surface.
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use futures_channel::oneshot;
use pen_protocol::{Catalog, RenderFrame, Viewport};
use pen_render::{Display, RenderTarget, Renderer, RendererOptions};
use web_sys::HtmlCanvasElement;

/// The canvas's surface, configured to the canvas's backing size.
pub struct Canvas {
    pub element: HtmlCanvasElement,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    frame: Option<wgpu::SurfaceTexture>,
    /// A backing size kept whatever the canvas shows (a replay's).
    pinned: Option<(u32, u32)>,
}

/// The backing size for `element`: its CSS size in device pixels (the
/// display's density and the page's zoom), each side at most `limit`.
fn backing(element: &HtmlCanvasElement, limit: u32) -> Option<(u32, u32)> {
    let ratio = web_sys::window().unwrap().device_pixel_ratio();
    let (width, height) = (element.client_width(), element.client_height());
    let css = (f64::from(width), f64::from(height));
    (width > 0 && height > 0).then(|| pen_runtime::scale::physical_size(css, ratio, limit))
}

impl Canvas {
    /// Backs the canvas with device pixels, as many as it shows now (the
    /// ratio changes as the page moves between screens or zooms), and the
    /// surface with the canvas.
    fn fit(&mut self) {
        let limit = self.device.limits().max_texture_dimension_2d;
        let backed = self.pinned.map(|(width, height)| (width.clamp(1, limit), height.clamp(1, limit)));
        if let Some((width, height)) = backed.or_else(|| backing(&self.element, limit)) {
            if self.element.width() != width {
                self.element.set_width(width);
            }
            if self.element.height() != height {
                self.element.set_height(height);
            }
        }
        let size = (self.element.width().max(1), self.element.height().max(1));
        if size != (self.config.width, self.config.height) {
            (self.config.width, self.config.height) = size;
            self.surface.configure(&self.device, &self.config);
        }
    }
}

impl RenderTarget for Canvas {
    fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    fn view(&mut self) -> Option<wgpu::TextureView> {
        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => {
                self.surface.configure(&self.device, &self.config);
                return None;
            }
        };
        let view = texture.texture.create_view(&Default::default());
        self.frame = Some(texture);
        Some(view)
    }

    // Keep the surface until the dev tools have drawn over the game.
    fn present(&mut self) {}
}

pub struct Screen {
    /// The graphics adapter's name, and whether it is WebGPU's or WebGL2's.
    pub adapter: String,
    pub renderer: Renderer,
    /// The dev tools' sprites, over the game's frame.
    pub overlay: Renderer,
    pub canvas: Canvas,
    /// The GPU finished the last frame submitted: the next waits for it.
    in_flight: Option<oneshot::Receiver<()>>,
    /// Drawing through WebGL2, wgpu's GL backend.
    webgl: bool,
}

impl Screen {
    pub async fn new(element: HtmlCanvasElement, assets: String) -> Result<Self> {
        let veil = Some(0.6);
        // WebGPU if the browser can give an adapter, else WebGL2. That is
        // settled before the canvas has a context: one asked for "webgpu"
        // cannot give "webgl2".
        let instance = wgpu::util::new_instance_with_webgpu_detection(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        })
        .await;
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(element.clone()))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await?;
        let info = adapter.get_info();
        let webgl = info.backend == wgpu::Backend::Gl;
        let descriptor = if webgl {
            // The scene draws in floating point, and decals read a float
            // copy of its depth (pen-render's downlevel.rs).
            for format in [wgpu::TextureFormat::Rgba16Float, wgpu::TextureFormat::R32Float] {
                let usages = adapter.get_texture_format_features(format).allowed_usages;
                ensure!(
                    usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT),
                    "this browser's WebGL2 cannot draw into {format:?} textures (EXT_color_buffer_float), and it has \
                     no WebGPU"
                );
            }
            // WebGL2 meets few of WebGPU's default limits: what it has. It
            // has no storage buffers, which pen-render draws without.
            wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() }
        } else {
            wgpu::DeviceDescriptor::default()
        };
        // With the texture compression the adapter offers, and timestamps,
        // which time the GPU's passes for the dev tools, where it has them.
        let descriptor =
            wgpu::DeviceDescriptor { required_features: pen_render::optional_features(&adapter), ..descriptor };
        let (device, queue) = adapter.request_device(&descriptor).await?;
        let api = if webgl { "WebGL2" } else { "WebGPU" };
        // The API first: the diagnostics cut long names short.
        let name = if info.name.is_empty() { api.to_owned() } else { format!("{api}: {}", info.name) };
        // Once each: an error a frame repeats would otherwise flood the console.
        let reported = Mutex::new(HashSet::new());
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            let message = error.to_string();
            if reported.lock().unwrap().insert(message.clone()) {
                web_sys::console::error_1(&format!("GPU error: {message}").into());
            }
        }));
        // The browser's preferred format, which is not sRGB: the present
        // pass encodes colours itself.
        let capabilities = surface.get_capabilities(&adapter);
        let format =
            capabilities.formats.iter().copied().find(|format| !format.is_srgb()).unwrap_or(capabilities.formats[0]);
        let alpha_mode = if capabilities.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            capabilities.alpha_modes[0]
        };
        if let Some((width, height)) = backing(&element, device.limits().max_texture_dimension_2d) {
            element.set_width(width);
            element.set_height(height);
        }
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: element.width().max(1),
            height: element.height().max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: Vec::new(),
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);
        let options = RendererOptions {
            format,
            display: Display::SDR,
            assets: assets.clone().into(),
            veil,
            pipeline_cache: None,
        };
        let renderer = Renderer::new(device.clone(), queue.clone(), options)?;
        let overlay = renderer.share(assets.into())?;
        let canvas = Canvas { element, queue, surface, config, device, frame: None, pinned: None };
        Ok(Self { adapter: name, renderer, overlay, canvas, in_flight: None, webgl })
    }

    /// Follows the canvas's size and the display's density: the backing
    /// size the next frame is laid out on and drawn into.
    pub fn fit(&mut self) {
        self.canvas.fit();
    }

    /// Keeps the canvas backed by `size` (a replay's viewport, which its
    /// frames were laid out on), shown whole within the canvas, or with
    /// None follows the canvas again.
    pub fn pin(&mut self, size: Option<(u32, u32)>) {
        self.canvas.pinned = size;
        let fit = if size.is_some() { "contain" } else { "" };
        let _ = self.canvas.element.style().set_property("object-fit", fit);
        self.canvas.fit();
    }

    pub fn viewport(&self) -> Viewport {
        Viewport { width: self.canvas.element.width().max(1), height: self.canvas.element.height().max(1) }
    }

    /// Draws the game's frame and the dev tools' over it. One frame is in
    /// flight: the GPU draws it while the next is computed, and the next
    /// waits for it before it is submitted, which bounds the queue on slow
    /// adapters (frame callbacks alone put no brake on WebGPU). WebGL2 bounds
    /// its own, and tells of work done only when polled after the page's
    /// task has ended: nothing waits on it there.
    pub async fn draw(&mut self, game: &RenderFrame, overlay: Option<&RenderFrame>, catalog: &Catalog) -> Result<()> {
        if let Some(previous) = self.in_flight.take() {
            let _ = previous.await;
        }
        self.canvas.fit();
        self.renderer.submit(game, catalog, &mut self.canvas)?;
        if let Some(overlay) = overlay {
            self.overlay.prepare_sprites(&overlay.sprites, catalog)?;
            if let Some(frame) = &self.canvas.frame {
                let view = frame.texture.create_view(&Default::default());
                self.overlay.overlay(&overlay.sprites, catalog, &view, self.canvas.size())?;
            }
        }
        if let Some(frame) = self.canvas.frame.take() {
            self.canvas.queue.present(frame);
        }
        if self.webgl {
            return Ok(());
        }
        let (done, finished) = oneshot::channel();
        self.canvas.queue.on_submitted_work_done(move || {
            let _ = done.send(());
        });
        self.in_flight = Some(finished);
        Ok(())
    }

    /// Whether assets are still loading, streaming in or fading in.
    pub fn busy(&self) -> bool {
        !self.renderer.quiet() || self.renderer.pending() > 0 || self.overlay.pending() > 0
    }
}
