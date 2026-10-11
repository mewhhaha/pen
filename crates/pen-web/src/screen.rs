use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use anyhow::Result;
use pen_protocol::{Catalog, RenderFrame, Viewport};
use pen_render::{RenderTarget, Renderer, RendererOptions};
use web_sys::HtmlCanvasElement;
use pen_runtime::game::Host;

/// The canvas's surface, configured to the canvas's size.
pub struct Canvas {
    pub element: HtmlCanvasElement,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    frame: Option<wgpu::SurfaceTexture>,
}

impl Canvas {
    /// Follows the canvas's size, which the page sets.
    fn fit(&mut self) {
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


pub struct Screen { pub renderer: Renderer, pub overlay: Renderer, pub canvas: Canvas }
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
        .await
        ?;
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await?;
    // Once each: an error a frame repeats would otherwise flood the console.
    let reported = Mutex::new(HashSet::new());
    device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
        let message = error.to_string();
        if reported.lock().unwrap().insert(message.clone()) {
            web_sys::console::error_1(&format!("GPU error: {message}").into());
        }
    }));
    // The browser's preferred format, which is not sRGB: the present pass
    // encodes as the TS's does.
    let capabilities = surface.get_capabilities(&adapter);
    let format = capabilities.formats.iter().copied().find(|format| !format.is_srgb()).unwrap_or(capabilities.formats[0]);
    let alpha_mode = if capabilities.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
        wgpu::CompositeAlphaMode::Opaque
    } else {
        capabilities.alpha_modes[0]
    };
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
    let renderer =
        Renderer::new(device.clone(), queue.clone(), RendererOptions { format, assets: assets.clone().into(), veil })?;

        let overlay = Renderer::new(device.clone(), queue.clone(), RendererOptions { format, assets: assets.into(), veil: None })?;
        Ok(Self { renderer, overlay, canvas: Canvas { element, queue, surface, config, device, frame: None } })
    }
    pub async fn settled(&self) {
        let (sent, received) = futures_channel::oneshot::channel();
        self.canvas.queue.on_submitted_work_done(move || { let _ = sent.send(()); });
        let _ = received.await;
    }
    pub fn draw(&mut self, game: &RenderFrame, overlay: Option<&RenderFrame>, catalog: &Catalog) -> Result<()> {
        self.canvas.fit();
        self.renderer.submit(game, catalog, &mut self.canvas)?;
        if let Some(overlay) = overlay {
            self.overlay.prepare_sprites(&overlay.sprites, catalog)?;
            if let Some(frame) = &self.canvas.frame {
                let view = frame.texture.create_view(&Default::default());
                self.overlay.overlay(&overlay.sprites, catalog, &view, self.canvas.size())?;
            }
        }
        if let Some(frame) = self.canvas.frame.take() { self.canvas.queue.present(frame); }
        Ok(())
    }
}
impl Host for Screen {
    fn viewport(&self) -> Viewport { Viewport { width: self.canvas.element.width(), height: self.canvas.element.height() } }
    fn validate(&self, frame: &RenderFrame) -> Result<()> { pen_render::validate_frame(frame) }
    fn render(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<()> { self.draw(frame, None, catalog) }
    fn report(&mut self, message: &str) { web_sys::console::log_1(&message.into()); }
}
