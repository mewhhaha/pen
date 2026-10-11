//! gdev's renderer in the browser: pen-render on wgpu's WebGPU backend,
//! drawing render packets into a canvas, with an API shaped like
//! renderer.ts's `createRenderer`:
//!
//! ```js
//! import init, { createRenderer } from "./pen_render.js";
//! await init();
//! const screen = await createRenderer({ canvas, assets: "assets", veil: 0.6 });
//! screen.preload(catalog);                  // catalog.json's object
//! screen.submit(packet, catalog, schema);   // the game's Float32Array packet
//! screen.takeEvents();                      // AssetReady/AssetFailed, as GameEvents
//! screen.quiet(); screen.pending(); screen.stats(); screen.framesInFlight();
//! screen.reload(["crater.wgsl"]);
//! ```
//!
//! It makes its own WebGPU device and configures the canvas with it.
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use js_sys::{Array, BigInt, Float32Array, Object, Reflect};
use pen_protocol::boundary::{decode_frame, parse_catalog};
use pen_protocol::{Catalog, GameEvent};
use pen_render::{RenderTarget, Renderer, RendererOptions};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

fn error(message: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&message.to_string()).into()
}

fn set(object: &Object, key: &str, value: impl Into<JsValue>) {
    let _ = Reflect::set(object, &key.into(), &value.into());
}

/// catalog.json's object, as the native host reads it.
fn catalog_of(value: &JsValue) -> Result<Catalog, JsValue> {
    let text = js_sys::JSON::stringify(value)?.as_string().unwrap_or_default();
    let json: serde_json::Value = serde_json::from_str(&text).map_err(error)?;
    parse_catalog(&json).map_err(error)
}

/// The canvas's surface, configured to the canvas's size.
struct Canvas {
    element: HtmlCanvasElement,
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

    fn present(&mut self) {
        if let Some(frame) = self.frame.take() {
            self.queue.present(frame);
        }
    }
}

#[wasm_bindgen]
pub struct PenRenderer {
    renderer: Renderer,
    canvas: Canvas,
    queue: wgpu::Queue,
    /// The catalog object frames were last decoded with, and what it says.
    catalog: Option<(JsValue, Catalog)>,
    /// Frames submitted that the GPU has not finished.
    in_flight: Arc<AtomicU32>,
}

/// `createRenderer({ canvas, assets, veil })`: `assets` is the URL the asset
/// keys are under (default "assets"), `veil` the seconds the loading veil
/// takes to clear (none without it).
#[wasm_bindgen(js_name = createRenderer)]
pub async fn create_renderer(options: JsValue) -> Result<PenRenderer, JsValue> {
    std::panic::set_hook(Box::new(|info| web_sys::console::error_1(&info.to_string().into())));
    let element: HtmlCanvasElement = Reflect::get(&options, &"canvas".into())?
        .dyn_into()
        .map_err(|_| error("createRenderer needs a canvas"))?;
    let assets = Reflect::get(&options, &"assets".into())?.as_string().unwrap_or_else(|| "assets".into());
    let veil = Reflect::get(&options, &"veil".into())?.as_f64().map(|seconds| seconds as f32);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(element.clone())).map_err(error)?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(error)?;
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.map_err(error)?;
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
        Renderer::new(device.clone(), queue.clone(), RendererOptions { format, assets: assets.into(), veil }).map_err(error)?;
    Ok(PenRenderer {
        renderer,
        canvas: Canvas { element, queue: queue.clone(), surface, config, device, frame: None },
        queue,
        catalog: None,
        in_flight: Arc::new(AtomicU32::new(0)),
    })
}

#[wasm_bindgen]
impl PenRenderer {
    /// Loads the catalog's meshes and materials, and their textures and
    /// shaders, ahead of need.
    pub fn preload(&mut self, catalog: JsValue) -> Result<(), JsValue> {
        let parsed = catalog_of(&catalog)?;
        self.renderer.preload(&parsed);
        self.catalog = Some((catalog, parsed));
        Ok(())
    }

    /// Decodes the game's render packet with `catalog` and `schema`,
    /// validates it and draws it into the canvas.
    pub fn submit(&mut self, packet: &Float32Array, catalog: JsValue, schema: u32) -> Result<(), JsValue> {
        let known = self.catalog.as_ref().is_some_and(|(object, _)| Object::is(object, &catalog));
        if !known {
            let parsed = catalog_of(&catalog)?;
            self.catalog = Some((catalog, parsed));
        }
        let catalog = &self.catalog.as_ref().unwrap().1;
        let decoded = decode_frame(&packet.to_vec(), schema, catalog).map_err(error)?;
        self.canvas.fit();
        self.renderer.submit(&decoded.render, catalog, &mut self.canvas).map_err(error)?;
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let in_flight = self.in_flight.clone();
        self.queue.on_submitted_work_done(move || {
            in_flight.fetch_sub(1, Ordering::Relaxed);
        });
        Ok(())
    }

    /// The asset events since the last call, as protocol.ts's GameEvents.
    #[wasm_bindgen(js_name = takeEvents)]
    pub fn take_events(&mut self) -> Array {
        let events = Array::new();
        for event in self.renderer.take_events() {
            let (tag, kind, key, revision, message) = match event {
                GameEvent::AssetReady { key, kind, revision } => ("AssetReady", kind, key, revision, None),
                GameEvent::AssetFailed { key, kind, revision, message } => ("AssetFailed", kind, key, revision, Some(message)),
                _ => continue,
            };
            let value = Object::new();
            set(&value, "kind", kind);
            set(&value, "key", key);
            set(&value, "revision", BigInt::from(revision));
            if let Some(message) = message {
                set(&value, "message", message);
            }
            let event = Object::new();
            set(&event, "tag", tag);
            set(&event, "value", value);
            events.push(&event);
        }
        events
    }

    /// The files of these asset keys changed: what of them was loaded loads again.
    pub fn reload(&mut self, keys: Vec<String>) {
        self.renderer.reload(&keys);
    }

    /// Whether nothing is streaming in or settling, and the veil is gone.
    pub fn quiet(&self) -> bool {
        self.renderer.quiet()
    }

    /// Asset work under way: loads, textures streaming in, compiles.
    pub fn pending(&self) -> u32 {
        self.renderer.pending() as u32
    }

    /// What the last frame drew, as renderer.ts's `stats()`.
    pub fn stats(&self) -> JsValue {
        let stats = self.renderer.stats();
        let object = Object::new();
        set(&object, "drawCalls", stats.draw_calls);
        set(&object, "instancedDrawCalls", stats.instanced_draw_calls);
        set(&object, "instances", stats.instances);
        object.into()
    }

    /// Frames submitted that the GPU has not finished drawing.
    #[wasm_bindgen(js_name = framesInFlight)]
    pub fn frames_in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::Relaxed)
    }
}
