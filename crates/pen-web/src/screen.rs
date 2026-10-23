//! The canvas on WebGPU: the game's renderer, and the dev tools' sharing its
//! device, drawing into the canvas's surface.
use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use futures_channel::oneshot;
use js_sys::Promise;
use pen_protocol::{Catalog, RenderFrame, Viewport};
use pen_render::{Display, RenderTarget, Renderer, RendererOptions};
use pen_runtime::game::DrawTimings;
use wasm_bindgen::{JsCast, prelude::*};
use wasm_bindgen_futures::JsFuture;
use web_sys::HtmlCanvasElement;
use web_time::Instant;

use crate::program::error;

/// Keep submission bounded without making each frame wait for its predecessor.
pub const FRAMES_IN_FLIGHT: usize = 2;

/// Resume immediately before the browser repaints, after all asynchronous work.
async fn next_frame() -> Result<f64> {
    let promise = Promise::new(&mut |resolve, reject| {
        let callback = Closure::once_into_js(move |time: f64| {
            let _ = resolve.call1(&JsValue::NULL, &time.into());
        });
        if let Err(error) = web_sys::window().unwrap().request_animation_frame(callback.unchecked_ref()) {
            let _ = reject.call1(&JsValue::NULL, &error);
        }
    });
    Ok(JsFuture::from(promise).await.map_err(error)?.as_f64().unwrap_or_default() / 1000.0)
}

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
    timings: DrawTimings,
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
        let started = Instant::now();
        let acquired = self.surface.get_current_texture();
        self.timings.acquire_ms += started.elapsed().as_secs_f64() * 1000.0;
        let texture = match acquired {
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
    /// Completion fences in submission order, at most FRAMES_IN_FLIGHT.
    in_flight: VecDeque<oneshot::Receiver<()>>,
}

impl Screen {
    pub async fn new(element: HtmlCanvasElement, assets: String) -> Result<Self> {
        let veil = Some(0.6);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let no_webgpu = "this browser has no WebGPU, which the game draws with";
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(element.clone())).context(no_webgpu)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .context(no_webgpu)?;
        let info = adapter.get_info();
        // With the texture compression the adapter offers, and timestamps,
        // which time the GPU's passes for the dev tools, where it has them.
        let descriptor =
            wgpu::DeviceDescriptor { required_features: pen_render::optional_features(&adapter), ..Default::default() };
        let (device, queue) = adapter.request_device(&descriptor).await?;
        // The API first: the diagnostics cut long names short.
        let name = if info.name.is_empty() { "WebGPU".to_owned() } else { format!("WebGPU: {}", info.name) };
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
        let canvas = Canvas {
            element,
            queue,
            surface,
            config,
            device,
            frame: None,
            pinned: None,
            timings: DrawTimings::default(),
        };
        Ok(Self { adapter: name, renderer, overlay, canvas, in_flight: VecDeque::new() })
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

    /// Draws in the browser's repaint callback, after guest work and GPU
    /// backpressure have finished. Two submissions may be in flight; a
    /// slow completion notification cannot serialize every frame.
    pub async fn draw(
        &mut self,
        game: &RenderFrame,
        overlay: Option<&RenderFrame>,
        catalog: &Catalog,
    ) -> Result<(f64, DrawTimings)> {
        self.canvas.timings = DrawTimings::default();
        let pacing = Instant::now();
        if self.in_flight.len() == FRAMES_IN_FLIGHT {
            let _ = self.in_flight.pop_front().unwrap().await;
        }
        let timestamp = next_frame().await?;
        self.canvas.timings.pace_ms = pacing.elapsed().as_secs_f64() * 1000.0;
        let rendering = Instant::now();
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
            let started = Instant::now();
            self.canvas.queue.present(frame);
            self.canvas.timings.present_ms += started.elapsed().as_secs_f64() * 1000.0;
        }
        let (done, finished) = oneshot::channel();
        self.canvas.queue.on_submitted_work_done(move || {
            let _ = done.send(());
        });
        self.in_flight.push_back(finished);
        let timings = &mut self.canvas.timings;
        timings.render_ms =
            (rendering.elapsed().as_secs_f64() * 1000.0 - timings.acquire_ms - timings.present_ms).max(0.0);
        Ok((timestamp, *timings))
    }

    /// Whether assets are still loading, streaming in or fading in.
    pub fn busy(&self) -> bool {
        !self.renderer.quiet() || self.renderer.pending() > 0 || self.overlay.pending() > 0
    }
}
