//! Browser front end for the shared pen runtime and WebGPU renderer. It
//! builds for wasm32-unknown-unknown only (`build.sh`).
#![cfg(target_arch = "wasm32")]
mod accessibility;
mod audio;
mod development;
mod files;
mod input;
mod preferences;
mod program;
mod replay;
mod run;
mod saves;
mod screen;
mod services;
mod tools;

use anyhow::Result;
use js_sys::Uint8Array;
use pen_protocol::{Catalog, GameEvent, RenderFrame, Viewport, service, service_status};
use pen_runtime::devtools::{Log, log_source};
use pen_runtime::game::Host;
use pen_runtime::snapshot::{self, SavedState};
use pen_runtime::state::{self, CONFIG_FILE, StateAdapter};
use std::sync::Arc;
use wasm_bindgen::{JsCast, prelude::*};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{HtmlCanvasElement, Response};

use program::{BrowserBackend, error};
use run::{Browser, Options, Runtime};
use screen::Screen;
use tools::Tools;

/// Fetches `path`, relative to the page.
pub async fn fetch(path: &str) -> Result<Vec<u8>> {
    let response: Response = JsFuture::from(web_sys::window().unwrap().fetch_with_str(path))
        .await
        .map_err(error)?
        .dyn_into()
        .map_err(error)?;
    anyhow::ensure!(response.ok(), "fetch {path}: HTTP {}", response.status());
    let bytes = JsFuture::from(response.array_buffer().map_err(error)?).await.map_err(error)?;
    Ok(Uint8Array::new(&bytes).to_vec())
}

/// Missing optional declarations are opaque; all other failures stay visible.
pub(crate) async fn load_state_adapter(assets: &str) -> Arc<dyn StateAdapter> {
    let path = format!("{assets}/{CONFIG_FILE}");
    let loaded: Result<Arc<dyn StateAdapter>> = async {
        let response: Response = JsFuture::from(web_sys::window().unwrap().fetch_with_str(&path))
            .await
            .map_err(error)?
            .dyn_into()
            .map_err(error)?;
        if response.status() == 404 {
            return Ok(state::opaque());
        }
        anyhow::ensure!(response.ok(), "HTTP {}", response.status());
        let bytes = JsFuture::from(response.array_buffer().map_err(error)?).await.map_err(error)?;
        let bytes = Uint8Array::new(&bytes).to_vec();
        pen_ink::adapter(std::str::from_utf8(&bytes)?)
    }
    .await;
    loaded.unwrap_or_else(|error| {
        let message = format!("{path}: {error:#}");
        web_sys::console::error_1(&message.clone().into());
        state::invalid(message)
    })
}

/// Names the page after the game, as its `game.json` (`pen build --web`
/// writes it) titles it; returns its name, which the files the page hands
/// the player are named after.
async fn manifest(manifest: &str, canvas: &HtmlCanvasElement) -> String {
    let game = fetch(manifest).await.ok().and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let field = |key: &str| {
        let value = game.as_ref()?.get(key)?.as_str()?;
        (!value.is_empty()).then(|| value.to_owned())
    };
    if let Some(title) = field("title") {
        web_sys::window().unwrap().document().unwrap().set_title(&title);
        let _ = canvas.set_attribute("aria-label", &title);
    }
    field("name").unwrap_or_else(|| "game".to_owned())
}

fn failed(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&format!("{error:#}"))
}

/// What the game's runtime needs while it starts: the viewport, checks,
/// and the log for what it made of the saved state.
struct Starting<'a> {
    viewport: Viewport,
    log: &'a mut Log,
    state_adapter: &'a Arc<dyn StateAdapter>,
}

impl Host for Starting<'_> {
    fn viewport(&self) -> Viewport {
        self.viewport
    }

    fn validate(&self, frame: &RenderFrame) -> Result<()> {
        pen_render::validate_frame(frame)
    }

    fn render(&mut self, _frame: &RenderFrame, _catalog: &Catalog) -> Result<()> {
        Ok(())
    }

    fn report(&mut self, message: &str) {
        run::say(self.log, log_source::GAME, false, message);
    }

    fn state_adapter(&self) -> Arc<dyn StateAdapter> {
        self.state_adapter.clone()
    }
}

/// Starts the game on `canvas`. `options` may name the files (`game`,
/// `devtools`, `catalog`, `scene`, `manifest` and the `assets` directory),
/// all relative to the page for portable exports, and set `development`
/// (served by `pen web`) and `qualify` (the browser harnesses' hooks). A
/// recording the page's link names (`#replay=` or `?replay=`) plays first.
#[wasm_bindgen]
pub async fn start(canvas: HtmlCanvasElement, options: JsValue) -> Result<(), JsValue> {
    std::panic::set_hook(Box::new(|info| web_sys::console::error_1(&info.to_string().into())));
    let path = |key: &str, default: &str| {
        js_sys::Reflect::get(&options, &key.into())
            .ok()
            .and_then(|value| value.as_string())
            .unwrap_or_else(|| default.into())
    };
    let flag =
        |key: &str| js_sys::Reflect::get(&options, &key.into()).ok().and_then(|value| value.as_bool()).unwrap_or(false);
    let name = manifest(&path("manifest", "game.json"), &canvas).await;
    let game = fetch(&path("game", "game.wasm")).await.map_err(failed)?;
    let catalog: serde_json::Value =
        serde_json::from_slice(&fetch(&path("catalog", "catalog.json")).await.map_err(failed)?).map_err(failed)?;
    let catalog = pen_protocol::boundary::parse_catalog(&catalog).map_err(failed)?;
    let scene = fetch(&path("scene", "scene.bin")).await.ok().and_then(|bytes| snapshot::decode(&bytes).ok());
    let devtools = fetch(&path("devtools", "devtools.wasm")).await.ok();
    let assets = path("assets", "assets");
    let state_adapter = load_state_adapter(&assets).await;
    // The sounds' captions, if the assets have any.
    let captions = match fetch(&format!("{assets}/captions.json")).await {
        Ok(bytes) => pen_runtime::captions::Captions::parse(&String::from_utf8_lossy(&bytes)).unwrap_or_else(|error| {
            web_sys::console::warn_1(&format!("{error:#}").into());
            Default::default()
        }),
        Err(_) => Default::default(),
    };
    let options = Options {
        development: flag("development"),
        qualify: flag("qualify"),
        name,
        replay: replay::linked(),
        state_adapter: state_adapter.clone(),
        assets: assets.clone(),
        captions,
    };
    let mut screen = Screen::new(canvas.clone(), assets.clone()).await.map_err(failed)?;
    screen.renderer.preload(&catalog);
    let mut log = Log::new(run::LOG_LINES);
    log.push(log_source::SESSION, false, "pen browser host ready");
    let viewport = screen.viewport();
    let started = {
        let mut host = Starting { viewport, log: &mut log, state_adapter: &state_adapter };
        Runtime::new(&BrowserBackend, game, catalog.clone(), &mut host, saves::read("pen.autosave").await, scene).await
    };
    let runtime = started.map_err(failed)?;
    let tools = match devtools {
        Some(module) => {
            let saved = saves::read("pen.devtools").await;
            Some(Tools::start(module, &catalog, viewport, saved, &mut log).await.map_err(failed)?)
        }
        None => None,
    };
    let input = input::Input::install(&canvas);
    let audio = audio::Audio::new(&canvas, assets);
    let browser = Browser::new(runtime, tools, screen, input, audio, log, options).map_err(failed)?;
    spawn_local(async move {
        if let Err(error) = browser.run().await {
            web_sys::console::error_1(&format!("{error:#}").into());
            if let Some(element) = web_sys::window().unwrap().document().unwrap().get_element_by_id("status") {
                element.set_text_content(Some(&format!("{error:#}")));
            }
        }
    });
    Ok(())
}

/// Browser engine qualification: identical request packets to `pen packets`.
/// It deliberately skips rendering and platform IO, just as native packet runs do.
#[wasm_bindgen]
pub async fn qualify(module: Uint8Array, scene: Uint8Array, inputs: js_sys::Array) -> Result<js_sys::Array, JsValue> {
    use pen_runtime::program::{Backend, Operation, Program};
    let result = async {
        let mut program = BrowserBackend.start(&module.to_vec()).await?;
        program.request(Operation::Create, &[]).await?;
        program.accept();
        if let Ok(scene) = snapshot::decode(&scene.to_vec())
            && scene.schema == program.schema()
        {
            program.request(Operation::LoadScene, &scene.state).await?;
            program.accept();
        }
        let output = js_sys::Array::new();
        for input in inputs.iter() {
            let packet = program.request(Operation::Frame, &js_sys::Float32Array::new(&input).to_vec()).await?;
            program.accept();
            output.push(&js_sys::Float32Array::from(packet.as_slice()));
        }
        let saved = SavedState { schema: program.schema(), state: program.request(Operation::Snapshot, &[]).await? };
        output.push(&Uint8Array::from(snapshot::encode(&saved)?.as_slice()));
        program.close().await;
        Ok::<_, anyhow::Error>(output)
    }
    .await;
    result.map_err(failed)
}

/// Exercises real browser platform adapters without changing a game's saves.
#[wasm_bindgen]
pub async fn qualify_platform() -> Result<String, JsValue> {
    let result = async {
        let saved = SavedState { schema: 5, state: vec![1.0, -2.5, 3.0] };
        saves::write("pen.qualification", &saved).await?;
        anyhow::ensure!(saves::read("pen.qualification").await == Some(saved), "browser snapshot round trip failed");
        let mut services = services::standard();
        let path = pen_protocol::ServiceRequest {
            request: 1,
            service: service::PATHFINDING,
            operation: pen_protocol::pathfinding_op::FIND,
            a: 3.0,
            b: 2.0,
            c: 0.0,
            d: 5.0,
            value_count: 6,
            values: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        };
        let steam = pen_protocol::ServiceRequest { request: 2, service: service::STEAM, ..Default::default() };
        let unknown = pen_protocol::ServiceRequest { request: 3, service: 999, ..Default::default() };
        services.answer(&[path, steam, unknown]);
        let answers = services.take_answers();
        let path = pen_runtime::services::Answer {
            values: vec![0.0, 1.0, 4.0, 5.0],
            ..pen_runtime::services::Answer::ok(4.0)
        };
        anyhow::ensure!(answers[..3] == path.events(1)[..], "browser pathfinding failed");
        let status = |request: u32| {
            answers.iter().find_map(|event| match event {
                GameEvent::ServiceAnswer { request: answered, status, .. } if *answered == request => Some(*status),
                _ => None,
            })
        };
        anyhow::ensure!(status(2) == Some(service_status::UNAVAILABLE), "browser Steam must be unavailable");
        anyhow::ensure!(status(3) == Some(service_status::UNKNOWN), "unregistered browser service must be unknown");
        pen_simulation::conformance::check(&mut services::standard()).map_err(anyhow::Error::msg)?;
        audio::qualify().await?;
        Ok::<_, anyhow::Error>(
            "IndexedDB snapshot, pathfinding, simulation services, service statuses and WebAudio graph passed"
                .to_owned(),
        )
    }
    .await;
    result.map_err(failed)
}

/// The simulation services' answers to fixed requests, here in wasm32, as
/// their digest: the same bits as native's, or an error saying what
/// differs. Needs no page, so node runs it too (tests/services.mjs).
#[wasm_bindgen]
pub fn qualify_services() -> Result<String, JsValue> {
    let mut services = pen_runtime::services::Services::<()>::new();
    pen_simulation::register(&mut services);
    pen_simulation::conformance::check(&mut services).map_err(failed)
}

/// Checks the host boundary with the same binary fixtures as the native tests.
#[wasm_bindgen]
pub fn qualify_frame(packet: &[f32], schema: u32, catalog: &str) -> Result<String, JsValue> {
    let catalog = serde_json::from_str(catalog).map_err(failed)?;
    let catalog = pen_protocol::boundary::parse_catalog(&catalog).map_err(failed)?;
    Ok(match pen_protocol::boundary::decode_frame(packet, schema, &catalog) {
        Ok(_) => String::new(),
        Err(error) => format!("{error:#}"),
    })
}
