//! Browser front end for the shared pen runtime and WebGPU renderer. It
//! builds for wasm32-unknown-unknown (`build.sh`), while exposing pure
//! target-independent browser host policies in [`policy`] for native unit tests.
pub mod policy;

#[cfg(target_arch = "wasm32")]
mod accessibility;
#[cfg(all(target_arch = "wasm32", feature = "audio-advanced"))]
mod audio;
#[cfg(all(target_arch = "wasm32", not(feature = "audio-advanced")))]
#[path = "audio_lite.rs"]
mod audio;
#[cfg(all(target_arch = "wasm32", feature = "devtools"))]
mod development;
#[cfg(target_arch = "wasm32")]
mod files;
#[cfg(target_arch = "wasm32")]
mod input;
#[cfg(all(target_arch = "wasm32", feature = "kernels"))]
mod kernels;
#[cfg(all(target_arch = "wasm32", feature = "netplay"))]
pub mod net;
#[cfg(target_arch = "wasm32")]
mod preferences;
#[cfg(target_arch = "wasm32")]
mod program;
#[cfg(all(target_arch = "wasm32", feature = "replay"))]
mod replay;
#[cfg(target_arch = "wasm32")]
mod run;
#[cfg(target_arch = "wasm32")]
mod saves;
#[cfg(target_arch = "wasm32")]
mod screen;
#[cfg(target_arch = "wasm32")]
mod services;
#[cfg(all(target_arch = "wasm32", feature = "devtools"))]
mod tools;
#[cfg(all(target_arch = "wasm32", feature = "video"))]
mod video;

#[cfg(target_arch = "wasm32")]
use anyhow::Result;
#[cfg(target_arch = "wasm32")]
use js_sys::Uint8Array;
#[cfg(target_arch = "wasm32")]
use pen_protocol::{Catalog, RenderFrame, Viewport};
#[cfg(all(target_arch = "wasm32", feature = "devtools"))]
use pen_protocol::{GameEvent, service, service_status};
#[cfg(target_arch = "wasm32")]
use pen_runtime::devtools::{Log, log_source};
#[cfg(target_arch = "wasm32")]
use pen_runtime::game::Host;
#[cfg(target_arch = "wasm32")]
use pen_runtime::snapshot;
#[cfg(all(target_arch = "wasm32", feature = "devtools"))]
use pen_runtime::snapshot::SavedState;
#[cfg(all(target_arch = "wasm32", feature = "saved-adapter"))]
use pen_runtime::state::CONFIG_FILE;
#[cfg(target_arch = "wasm32")]
use pen_runtime::state::{self, StateAdapter};
#[cfg(target_arch = "wasm32")]
use pen_runtime::web::WebFiles;
#[cfg(target_arch = "wasm32")]
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, prelude::*};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{JsFuture, spawn_local};
#[cfg(target_arch = "wasm32")]
use web_sys::{HtmlCanvasElement, Response};

#[cfg(target_arch = "wasm32")]
use program::{BrowserBackend, error};
#[cfg(target_arch = "wasm32")]
use run::{Browser, Options, Runtime};
#[cfg(target_arch = "wasm32")]
use screen::Screen;
#[cfg(all(target_arch = "wasm32", feature = "devtools"))]
use tools::Tools;

/// Fetches `path`, relative to the page.
#[cfg(target_arch = "wasm32")]
pub async fn fetch(path: &str) -> Result<Vec<u8>> {
    read_response(path, get_response(path).await?).await
}

#[cfg(target_arch = "wasm32")]
async fn get_response(path: &str) -> Result<Response> {
    JsFuture::from(web_sys::window().unwrap().fetch_with_str(path)).await.map_err(error)?.dyn_into().map_err(error)
}

#[cfg(target_arch = "wasm32")]
async fn read_response(path: &str, response: Response) -> Result<Vec<u8>> {
    anyhow::ensure!(response.ok(), "fetch {path}: HTTP {}", response.status());
    let bytes = JsFuture::from(response.array_buffer().map_err(error)?).await.map_err(error)?;
    Ok(Uint8Array::new(&bytes).to_vec())
}

/// Loads the game's string tables through the translation index beside the page.
#[cfg(all(target_arch = "wasm32", feature = "locales"))]
async fn load_locales() -> Result<Option<pen_runtime::locale::Locales>> {
    use anyhow::Context;

    let response = get_response("text.json").await?;
    if response.status() == 404 {
        return Ok(None);
    }
    let base = read_response("text.json", response).await?;
    let base = std::str::from_utf8(&base).context("text.json")?;
    let warn = |error| web_sys::console::warn_1(&format!("the game's string tables: {error}").into());
    let tags = async {
        let response = get_response("locales/index.json").await?;
        if response.status() == 404 {
            return Ok(Vec::new());
        }
        let bytes = read_response("locales/index.json", response).await?;
        serde_json::from_slice::<Vec<String>>(&bytes).context("locales/index.json")
    }
    .await
    .unwrap_or_else(|error| {
        warn(format!("{error:#}"));
        Vec::new()
    });
    let mut texts = Vec::with_capacity(tags.len());
    for tag in &tags {
        let path = format!("locales/{}.json", js_sys::encode_uri_component(tag));
        let result: Result<String> = async { Ok(String::from_utf8(fetch(&path).await?)?) }.await;
        match result {
            Ok(text) => texts.push((tag.as_str(), text)),
            Err(error) => warn(format!("locales/{tag}.json: {error:#}")),
        }
    }
    let translations: Vec<_> = texts.iter().map(|(tag, json)| (*tag, json.as_str())).collect();
    let loaded = pen_runtime::locale::Locales::from_json(base, "en", &translations);
    if let Some(locales) = loaded.locales {
        for error in loaded.errors {
            warn(error.to_string());
        }
        Ok(Some(locales))
    } else {
        let mut errors = loaded.errors.into_iter();
        let base_error = errors.next().expect("missing base table has a load error");
        for error in errors {
            warn(error.to_string());
        }
        anyhow::bail!("{base_error}")
    }
}

/// Fetches `path`, rejecting responses whose `Content-Length` or payload exceeds `max_bytes`
/// before copying into Wasm linear memory.
#[cfg(all(target_arch = "wasm32", feature = "video"))]
pub async fn fetch_bounded(path: &str, max_bytes: usize) -> Result<Vec<u8>> {
    let response: Response = JsFuture::from(web_sys::window().unwrap().fetch_with_str(path))
        .await
        .map_err(error)?
        .dyn_into()
        .map_err(error)?;
    anyhow::ensure!(response.ok(), "fetch {path}: HTTP {}", response.status());
    if let Ok(Some(len_header)) = response.headers().get("content-length") {
        anyhow::ensure!(
            policy::content_length_within_limit(&len_header, max_bytes),
            "fetch {path}: Content-Length {len_header} exceeds {max_bytes}-byte limit"
        );
    }
    let buffer = JsFuture::from(response.array_buffer().map_err(error)?).await.map_err(error)?;
    let array = Uint8Array::new(&buffer);
    let len = array.byte_length() as usize;
    anyhow::ensure!(len <= max_bytes, "fetch {path}: payload {len} bytes exceeds {max_bytes}-byte limit");
    Ok(array.to_vec())
}

/// Missing optional declarations are opaque; all other failures stay visible.
#[cfg(all(target_arch = "wasm32", feature = "saved-adapter"))]
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

#[cfg(all(target_arch = "wasm32", not(feature = "saved-adapter")))]
#[cfg_attr(not(feature = "devtools"), allow(dead_code))]
pub(crate) async fn load_state_adapter(_assets: &str) -> Arc<dyn StateAdapter> {
    state::opaque()
}

/// Names the page after the game, as its `game.json` (`pen build --web`
/// writes it) titles it, and reads the exported list of optional files.
#[cfg(target_arch = "wasm32")]
async fn manifest(manifest: &str, canvas: &HtmlCanvasElement) -> (String, Option<WebFiles>) {
    let game = fetch(manifest).await.ok().and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let field = |key: &str| {
        let value = game.as_ref()?.get(key)?.as_str()?;
        (!value.is_empty()).then(|| value.to_owned())
    };
    if let Some(title) = field("title") {
        web_sys::window().unwrap().document().unwrap().set_title(&title);
        let _ = canvas.set_attribute("aria-label", &title);
    }
    let name = field("name").unwrap_or_else(|| "game".to_owned());
    let files = game.and_then(|game| serde_json::from_value(game.get("web")?.clone()).ok());
    (name, files)
}

#[cfg(target_arch = "wasm32")]
async fn optional_file(path: &str, present: bool) -> Option<Vec<u8>> {
    if present { fetch(path).await.ok() } else { None }
}

#[cfg(target_arch = "wasm32")]
fn failed(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&format!("{error:#}"))
}

/// What the game's runtime needs while it starts: the viewport, checks,
/// and the log for what it made of the saved state.
#[cfg(target_arch = "wasm32")]
struct Starting<'a> {
    viewport: Viewport,
    log: &'a mut Log,
    state_adapter: &'a Arc<dyn StateAdapter>,
}

#[cfg(target_arch = "wasm32")]
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
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn start(canvas: HtmlCanvasElement, options: JsValue) -> Result<(), JsValue> {
    std::panic::set_hook(Box::new(|info| web_sys::console::error_1(&info.to_string().into())));
    let string = |key: &str| js_sys::Reflect::get(&options, &key.into()).ok().and_then(|value| value.as_string());
    let path = |key: &str, default: &str| string(key).unwrap_or_else(|| default.into());
    let flag =
        |key: &str| js_sys::Reflect::get(&options, &key.into()).ok().and_then(|value| value.as_bool()).unwrap_or(false);
    let assets = path("assets", "assets");
    let development = flag("development");
    if !cfg!(feature = "devtools") && (development || flag("qualify")) {
        return Err(failed("use a browser host built with devtools for development or qualification"));
    }
    if !cfg!(feature = "netplay") && js_sys::Reflect::has(&options, &"netRelay".into()).unwrap_or(false) {
        return Err(failed("use a browser host built with netplay for netRelay"));
    }
    let manifest_path = path("manifest", "game.json");
    let game_path = path("game", "game.wasm");
    let catalog_path = path("catalog", "catalog.json");
    let scene_path = path("scene", "scene.bin");
    let devtools_path = path("devtools", "devtools.wasm");
    let captions_path = format!("{assets}/captions.json");
    let content = async {
        let (name, files) = manifest(&manifest_path, &canvas).await;
        // A development page reads the current checkout. Exports already know what they ship.
        let files = files.unwrap_or(WebFiles {
            scene: development || string("scene").is_some(),
            devtools: development || string("devtools").is_some(),
            saved: development,
            captions: development,
            locales: development,
        });
        if !cfg!(feature = "devtools") && files.devtools {
            return Err(failed("use a browser host built with devtools to run devtools.wasm"));
        }
        if !cfg!(feature = "saved-adapter") && files.saved {
            return Err(failed("use a browser host built with saved-adapter for assets/saved.json"));
        }
        if !cfg!(feature = "locales") && files.locales {
            return Err(failed("use a browser host built with locales for text.json"));
        }
        let saved = async { if files.saved { load_state_adapter(&assets).await } else { state::opaque() } };
        #[cfg(feature = "locales")]
        let locales = async {
            if !files.locales {
                return None;
            }
            load_locales().await.unwrap_or_else(|error| {
                web_sys::console::warn_1(&format!("the game's string tables: {error:#}").into());
                None
            })
        };
        #[cfg(not(feature = "locales"))]
        let locales = async { None::<()> };
        let (scene, devtools, state_adapter, captions, locales) = futures_util::join!(
            optional_file(&scene_path, files.scene),
            optional_file(&devtools_path, files.devtools),
            saved,
            optional_file(&captions_path, files.captions),
            locales,
        );
        Ok((name, scene, devtools, state_adapter, captions, locales))
    };
    let (game, catalog, screen, autosave, content) = futures_util::join!(
        fetch(&game_path),
        fetch(&catalog_path),
        Screen::new(canvas.clone(), assets.clone()),
        saves::read("pen.autosave"),
        content,
    );
    let game = game.map_err(failed)?;
    let catalog: serde_json::Value = serde_json::from_slice(&catalog.map_err(failed)?).map_err(failed)?;
    let catalog = pen_protocol::boundary::parse_catalog(&catalog).map_err(failed)?;
    let mut screen = screen.map_err(failed)?;
    screen.renderer.preload(&catalog);
    let (name, scene, devtools, state_adapter, captions, locales) = content?;
    #[cfg(not(feature = "locales"))]
    let _ = locales;
    let scene = scene.and_then(|bytes| snapshot::decode(&bytes).ok());
    let captions = captions.map_or_else(Default::default, |bytes| {
        pen_runtime::captions::Captions::parse(&String::from_utf8_lossy(&bytes)).unwrap_or_else(|error| {
            web_sys::console::warn_1(&format!("{error:#}").into());
            Default::default()
        })
    });
    #[cfg(feature = "netplay")]
    let relay_config =
        js_sys::Reflect::get(&options, &"netRelay".into()).ok().filter(|v| v.is_object()).and_then(|obj| {
            let url = js_sys::Reflect::get(&obj, &"url".into()).ok()?.as_string()?;
            let room = js_sys::Reflect::get(&obj, &"room".into()).ok()?.as_string()?;
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped positive JS netplay config"
            )]
            let (player, players, token) = (
                js_sys::Reflect::get(&obj, &"player".into()).ok()?.as_f64()?.clamp(0.0, 255.0) as usize,
                js_sys::Reflect::get(&obj, &"players".into()).ok()?.as_f64()?.clamp(1.0, 256.0) as usize,
                js_sys::Reflect::get(&obj, &"token".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .unwrap_or(0.0) as u64,
            );
            Some(run::NetRelayConfig { url, room, player, players, token })
        });
    let options = Options {
        development,
        qualify: flag("qualify"),
        name,
        #[cfg(feature = "replay")]
        replay: replay::linked(),
        state_adapter: state_adapter.clone(),
        assets: assets.clone(),
        captions,
        #[cfg(feature = "locales")]
        locales,
        #[cfg(feature = "netplay")]
        netplay: None,
    };
    let mut log = Log::new(run::LOG_LINES);
    log.push(log_source::SESSION, false, "pen browser host ready");
    let viewport = screen.viewport();
    #[cfg(feature = "kernels")]
    let game_wasm_for_kernels = game.clone();
    #[cfg(feature = "netplay")]
    let build_hash = pen_runtime::net::wire::hash_bytes(&game);
    let started = {
        let mut host = Starting { viewport, log: &mut log, state_adapter: &state_adapter };
        Runtime::new(&BrowserBackend, game, catalog.clone(), &mut host, autosave, scene).await
    };
    let runtime = started.map_err(failed)?;
    #[cfg(feature = "netplay")]
    let mut runtime = runtime;
    #[cfg(feature = "netplay")]
    let mut options = options;
    #[cfg(feature = "netplay")]
    if let Some(cfg) = relay_config {
        let link =
            net::BrowserRelayLink::connect(&cfg.url, &cfg.room, cfg.player, cfg.players, cfg.token).map_err(failed)?;
        let settings = pen_runtime::net::Settings::new(cfg.players, cfg.player, viewport);
        let start = runtime.moment().await.map_err(failed)?;
        options.netplay =
            Some(pen_runtime::net::Netplay::new(settings, Box::new(link), build_hash, &start, 1).map_err(failed)?);
    }
    #[cfg(feature = "devtools")]
    let tools = match devtools {
        Some(module) => {
            let saved = saves::read("pen.devtools").await;
            Some(Tools::start(module, &catalog, viewport, saved, &mut log).await.map_err(failed)?)
        }
        None => None,
    };
    #[cfg(not(feature = "devtools"))]
    let _ = devtools;
    #[cfg(feature = "kernels")]
    let kernels = kernels::WebKernelService::maybe_new(&game_wasm_for_kernels).await;
    let input = input::Input::install(&canvas);
    let audio = audio::Audio::new(&canvas, assets);
    let browser = Browser::new(
        runtime,
        #[cfg(feature = "kernels")]
        kernels,
        #[cfg(feature = "devtools")]
        tools,
        screen,
        input,
        audio,
        log,
        options,
    )
    .map_err(failed)?;
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
#[cfg(all(target_arch = "wasm32", feature = "devtools"))]
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
#[cfg(all(target_arch = "wasm32", feature = "devtools"))]
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
        let status = |request: u32| {
            answers.iter().find_map(|event| match event {
                GameEvent::ServiceAnswer { request: answered, status, .. } if *answered == request => Some(*status),
                _ => None,
            })
        };
        anyhow::ensure!(status(1) == Some(service_status::UNKNOWN), "host navigation must be unregistered");
        anyhow::ensure!(status(2) == Some(service_status::UNAVAILABLE), "browser Steam must be unavailable");
        anyhow::ensure!(status(3) == Some(service_status::UNKNOWN), "unregistered browser service must be unknown");
        pen_simulation::conformance::check(&mut services::standard()).map_err(anyhow::Error::msg)?;
        audio::qualify().await?;
        Ok::<_, anyhow::Error>(
            "IndexedDB snapshot, noise, retired navigation, service statuses and WebAudio graph passed".to_owned(),
        )
    }
    .await;
    result.map_err(failed)
}

/// The simulation services' answers to fixed requests, here in wasm32, as
/// their digest: the same bits as native's, or an error saying what
/// differs. Needs no page, so node runs it too (tests/services.mjs).
#[cfg(all(target_arch = "wasm32", feature = "devtools"))]
#[wasm_bindgen]
pub fn qualify_services() -> Result<String, JsValue> {
    let mut services = pen_runtime::services::Services::<()>::new();
    pen_simulation::register(&mut services);
    pen_simulation::conformance::check(&mut services).map_err(failed)
}

/// Checks the host boundary with the same binary fixtures as the native tests.
#[cfg(all(target_arch = "wasm32", feature = "devtools"))]
#[wasm_bindgen]
pub fn qualify_frame(packet: &[f32], schema: u32, catalog: &str) -> Result<String, JsValue> {
    let catalog = serde_json::from_str(catalog).map_err(failed)?;
    let catalog = pen_protocol::boundary::parse_catalog(&catalog).map_err(failed)?;
    Ok(match pen_protocol::boundary::decode_frame(packet, schema, &catalog) {
        Ok(_) => String::new(),
        Err(error) => format!("{error:#}"),
    })
}
