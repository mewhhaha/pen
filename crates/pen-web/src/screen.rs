//! The canvas on WebGPU: the game's renderer, and the dev tools' sharing its
//! device, drawing into the canvas's surface.
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures_channel::oneshot;
use pen_protocol::{Catalog, RenderFrame, Viewport};
use pen_render::{RenderTarget, Renderer, RendererOptions};
use web_sys::HtmlCanvasElement;

/// The canvas's surface, configured to the canvas's backing size.
pub struct Canvas {
    pub element: HtmlCanvasElement,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    frame: Option<wgpu::SurfaceTexture>,
}

/// The backing size for `element`: its CSS size in device pixels (the
/// display's density and the page's zoom), each side at most `limit`.
fn backing(element: &HtmlCanvasElement, limit: u32) -> Option<(u32, u32)> {
    let ratio = web_sys::window().unwrap().device_pixel_ratio();
    let (width, height) = (element.client_width(), element.client_height());
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "saturating, then clamped")]
    let side = |css: i32| ((f64::from(css) * ratio).round() as u32).clamp(1, limit);
    (width > 0 && height > 0).then(|| (side(width), side(height)))
}

impl Canvas {
    /// Backs the canvas with device pixels, as many as it shows now (the
    /// ratio changes as the page moves between screens or zooms), and the
    /// surface with the canvas.
    fn fit(&mut self) {
        if let Some((width, height)) = backing(&self.element, self.device.limits().max_texture_dimension_2d) {
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
    /// The graphics adapter's name.
    pub adapter: String,
    pub renderer: Renderer,
    /// The dev tools' sprites, over the game's frame.
    pub overlay: Renderer,
    pub canvas: Canvas,
    /// The GPU finished the last frame submitted: the next waits for it.
    in_flight: Option<oneshot::Receiver<()>>,
}

impl Screen {
    pub async fn new(element: HtmlCanvasElement, assets: String) -> Result<Self> {
        let veil = Some(0.6);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(element.clone()))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await?;
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await?;
        let info = adapter.get_info();
        let name = if info.name.is_empty() { "WebGPU".to_owned() } else { format!("{} (WebGPU)", info.name) };
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
        let options = RendererOptions { format, assets: assets.clone().into(), veil, pipeline_cache: None };
        let renderer = Renderer::new(device.clone(), queue.clone(), options)?;
        let overlay = renderer.share(assets.into())?;
        let canvas = Canvas { element, queue, surface, config, device, frame: None };
        Ok(Self { adapter: name, renderer, overlay, canvas, in_flight: None })
    }

    /// Follows the canvas's size and the display's density: the backing
    /// size the next frame is laid out on and drawn into.
    pub fn fit(&mut self) {
        self.canvas.fit();
    }

    pub fn viewport(&self) -> Viewport {
        Viewport { width: self.canvas.element.width().max(1), height: self.canvas.element.height().max(1) }
    }

    /// Draws the game's frame and the dev tools' over it. One frame is in
    /// flight: the GPU draws it while the next is computed, and the next
    /// waits for it before it is submitted, which bounds the queue on slow
    /// adapters (frame callbacks alone put no brake on WebGPU).
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
