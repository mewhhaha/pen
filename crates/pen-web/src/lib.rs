//! Browser front end for the shared pen runtime and WebGPU renderer.
mod input;
mod program;
mod screen;
mod services;
mod audio;
mod accessibility;
mod saves;

use std::collections::VecDeque;
use anyhow::Result;
use js_sys::{Promise, Uint8Array};
use pen_protocol::{Catalog, Frame, GameEvent, RenderFrame, Viewport, devtools_frame, devtools_op, service, ui_role};
use pen_runtime::game::{GameRuntime, Host, Moment};
use pen_runtime::services::Answer;
use pen_runtime::snapshot::{self, SavedState};
use wasm_bindgen::{JsCast, prelude::*};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{HtmlCanvasElement, Response};
use program::{BrowserBackend, error};
use screen::Screen;

type Runtime = GameRuntime<BrowserBackend>;

pub async fn fetch(path: &str) -> Result<Vec<u8>> {
    let response: Response = JsFuture::from(web_sys::window().unwrap().fetch_with_str(path)).await.map_err(error)?.dyn_into().map_err(error)?;
    anyhow::ensure!(response.ok(), "fetch {path}: HTTP {}", response.status());
    let bytes = JsFuture::from(response.array_buffer().map_err(error)?).await.map_err(error)?;
    Ok(Uint8Array::new(&bytes).to_vec())
}
fn storage() -> Option<web_sys::Storage> { web_sys::window()?.local_storage().ok().flatten() }
fn preferences() -> Vec<GameEvent> {
    let value = storage().and_then(|s| s.get_item("pen.preferences").ok().flatten()).unwrap_or_default();
    let values: Vec<f32> = serde_json::from_str(&value).unwrap_or_default();
    values.into_iter().enumerate().filter(|(_, value)| value.is_finite()).map(|(index, value)| GameEvent::Preference { index: index as f32, value }).collect()
}
async fn bake(tuned: &std::collections::HashMap<u32,f32>) -> Result<String> {
    let mut values: Vec<_> = tuned.iter().map(|(id,value)| (*id,*value)).collect();
    values.sort_by_key(|(id,_)| *id);
    let options = web_sys::RequestInit::new();
    options.set_method("POST");
    options.set_body(&serde_json::to_string(&values)?.into());
    let request = web_sys::Request::new_with_str_and_init("/__pen/bake", &options).map_err(error)?;
    request.headers().set("Content-Type", "application/json").map_err(error)?;
    let response: Response = JsFuture::from(web_sys::window().unwrap().fetch_with_request(&request)).await.map_err(error)?.dyn_into().map_err(error)?;
    let message = JsFuture::from(response.text().map_err(error)?).await.map_err(error)?.as_string().unwrap_or_default();
    anyhow::ensure!(response.ok(), "{message}");
    Ok(message)
}
async fn next_frame() -> Result<f64> {
    let promise = Promise::new(&mut |resolve, reject| {
        let callback = Closure::once_into_js(move |time: f64| { let _ = resolve.call1(&JsValue::NULL, &time.into()); });
        if let Err(error) = web_sys::window().unwrap().request_animation_frame(callback.unchecked_ref()) { let _ = reject.call1(&JsValue::NULL, &error); }
    });
    Ok(JsFuture::from(promise).await.map_err(error)?.as_f64().unwrap_or_default() / 1000.0)
}
struct Shell(Viewport);
impl Host for Shell {
    fn viewport(&self) -> Viewport { self.0 }
    fn validate(&self, frame: &RenderFrame) -> Result<()> { pen_render::validate_frame(frame) }
    fn render(&mut self, _: &RenderFrame, _: &Catalog) -> Result<()> { Ok(()) }
    fn report(&mut self, message: &str) { web_sys::console::log_1(&message.into()); }
}
fn over(frame: Option<&RenderFrame>, (x,y): (f32,f32)) -> bool {
    frame.is_some_and(|frame| frame.ui.iter().any(|node| node.role == ui_role::DIALOG && !node.hidden
        && x >= node.x && y >= node.y && x < node.x + node.width && y < node.y + node.height))
}

/// Starts a Rust host; all paths are relative to this page for portable exports.
#[wasm_bindgen]
pub async fn start(canvas: HtmlCanvasElement, options: JsValue) -> Result<(), JsValue> {
    std::panic::set_hook(Box::new(|info| web_sys::console::error_1(&info.to_string().into())));
    let path = |key: &str, default: &str| js_sys::Reflect::get(&options, &key.into()).ok().and_then(|s| s.as_string()).unwrap_or_else(|| default.into());
    let game = fetch(&path("game", "game.wasm")).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
    let catalog: serde_json::Value = serde_json::from_slice(&fetch(&path("catalog", "catalog.json")).await.map_err(|e| JsValue::from_str(&e.to_string()))?).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let catalog = pen_protocol::boundary::parse_catalog(&catalog).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let scene = fetch(&path("scene", "scene.bin")).await.ok().and_then(|b| snapshot::decode(&b).ok());
    let devtools = fetch(&path("devtools", "devtools.wasm")).await.ok();
    let assets = path("assets", "assets");
    let development = js_sys::Reflect::get(&options, &"development".into()).ok().and_then(|v| v.as_bool()).unwrap_or(false);
    let mut screen = Screen::new(canvas.clone(), assets.clone()).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
    screen.renderer.preload(&catalog);
    // The first submitted frame must include persisted quality preferences.
    let runtime = Runtime::new(&BrowserBackend, game, catalog.clone(), &mut Shell(screen.viewport()), saves::read("pen.autosave").await, scene).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
    let overlay = match devtools {
        Some(bytes) => Some(Runtime::new(&BrowserBackend, bytes, catalog.clone(), &mut Shell(screen.viewport()), saves::read("pen.devtools").await, None).await.map_err(|e| JsValue::from_str(&e.to_string()))?),
        None => None,
    };
    let input = input::Input::install(&canvas);
    let audio = audio::Audio::new(&canvas, assets);
    spawn_local(async move {
        let result = run(runtime, overlay, screen, input, audio, development).await;
        if let Err(error) = result {
            web_sys::console::error_1(&error.to_string().into());
            if let Some(element) = web_sys::window().unwrap().document().unwrap().get_element_by_id("status") {
                element.set_text_content(Some(&format!("{error:#}")));
            }
        }
    });
    Ok(())
}

async fn run(mut game: Runtime, mut tools: Option<Runtime>, mut screen: Screen,
    input: std::rc::Rc<std::cell::RefCell<input::Input>>, mut audio: audio::Audio, development: bool) -> Result<()> {
    let mut accessibility = accessibility::Accessibility::new(input.clone())?;
    let mut events = preferences();
    let mut tools_events = Vec::new();
    let mut overlay: Option<RenderFrame> = None;
    let mut shown = game.take_drawn().unwrap_or_default();
    let mut history: VecDeque<Moment> = VecDeque::new();
    let mut position = 0usize;
    let mut paused = false;
    let mut replaying = false;
    let mut stepping = 0u32;
    let mut frame_number = 0u64;
    let mut measuring = false;
    let mut submitted_ms = 0.0f32;
    let mut housekeeping_ms = 0.0f32;
    let mut previous = 0.0;
    let mut last_save = 0.0;
    let mut last_reload = 0.0;
    let mut revision = String::new();
    let mut logs: Vec<(u64, u32, bool, String)> = vec![(1, 5, false, "pen browser host ready".into())];
    let mut next_log = 2u64;
    let mut catalog = game.catalog().clone();
    loop {
        let timestamp = next_frame().await?;
        let interval = timestamp - previous;
        previous = timestamp;
        let window = web_sys::window().unwrap();
        let canvas = &screen.canvas.element;
        let width = window.inner_width().map_err(error)?.as_f64().unwrap_or(800.0) as u32;
        let height = window.inner_height().map_err(error)?.as_f64().unwrap_or(600.0) as u32;
        if canvas.width() != width { canvas.set_width(width); }
        if canvas.height() != height { canvas.set_height(height); }
        let viewport = screen.viewport();
        let (incoming, pointer, hotkeys) = {
            let mut input = input.borrow_mut();
            (std::mem::take(&mut input.events), input.pointer, std::mem::take(&mut input.hotkeys))
        };
        for hotkey in hotkeys {
            match hotkey {
                pen_runtime::input::Hotkey::Pause => paused = !paused,
                pen_runtime::input::Hotkey::Step => { paused = true; stepping += 1; }
                pen_runtime::input::Hotkey::Back => {
                    position = position.saturating_sub(1);
                    if let Some(moment) = history.get(position) { shown = game.restore(&mut screen, moment).await?; paused = true; events = preferences(); }
                }
                pen_runtime::input::Hotkey::Replay => {
                    position = position.saturating_sub(300); replaying = !history.is_empty(); paused = true;
                }
            }
        }
        let typing = overlay.as_ref().is_some_and(|frame| frame.ui.iter().any(|node| node.focused));
        let pointer_taken = over(overlay.as_ref(), pointer);
        tools_events.extend(incoming.iter().cloned());
        for event in incoming {
            let taken = match &event {
                GameEvent::KeyDown(key) | GameEvent::KeyUp(key) => key == "`" || typing,
                GameEvent::PointerDown { .. } | GameEvent::PointerUp { .. } | GameEvent::PointerMove { .. } | GameEvent::Wheel { .. } => pointer_taken,
                _ => false,
            };
            if !taken { events.push(event); }
        }
        tools_events.extend(screen.overlay.take_events());
        if let Some(runtime) = tools.as_mut() {
            let input = runtime.input(&Frame { timestamp, viewport, events: std::mem::take(&mut tools_events) })?;
            runtime.send(&input)?;
            let rendered = runtime.receive(&Shell(viewport)).await?;
            for request in &rendered.services {
                let answer = if request.service == service::DEVTOOLS {
                    match request.operation {
                        devtools_op::GPU => Answer::text("WebGPU"),
                        devtools_op::FRAME => {
                            if measuring != (request.a == 1.0) {
                                measuring = request.a == 1.0;
                                game.measure_frames(measuring);
                            }
                            let mut values = vec![0.0; devtools_frame::NAMES.len()];
                            for at in [devtools_frame::GUEST, devtools_frame::WAIT, devtools_frame::DECODE, devtools_frame::VALIDATE,
                                devtools_frame::SUBMIT, devtools_frame::HOUSEKEEPING, devtools_frame::TOTAL, devtools_frame::PACKET,
                                devtools_frame::RECORDING] { values[at as usize] = -1.0; }
                            values[devtools_frame::INTERVAL as usize] = (interval * 1000.0) as f32;
                            values[devtools_frame::POSITION as usize] = frame_number as f32;
                            values[devtools_frame::PAUSED as usize] = f32::from(u8::from(paused));
                            values[devtools_frame::REPLAYING as usize] = f32::from(u8::from(replaying));
                            values[devtools_frame::HISTORY as usize] = f32::from(u8::from(!history.is_empty()));
                            values[devtools_frame::MEMORY as usize] = game.memory_bytes() as f32;
                            values[devtools_frame::WIDTH as usize] = viewport.width as f32;
                            values[devtools_frame::HEIGHT as usize] = viewport.height as f32;
                            let stats = screen.renderer.stats();
                            values[devtools_frame::DRAW_CALLS as usize] = stats.draw_calls as f32;
                            values[devtools_frame::INSTANCED_DRAW_CALLS as usize] = stats.instanced_draw_calls as f32;
                            values[devtools_frame::INSTANCES as usize] = stats.instances as f32;
                            values[devtools_frame::MESHES as usize] = shown.draws.len() as f32;
                            values[devtools_frame::BRUSHES as usize] = shown.solids.len() as f32;
                            if let Some(t) = game.last_timings() {
                                values[devtools_frame::GUEST as usize] = t.guest_ms as f32;
                                values[devtools_frame::WAIT as usize] = t.wait_ms as f32;
                                values[devtools_frame::DECODE as usize] = t.decode_ms as f32;
                                values[devtools_frame::VALIDATE as usize] = t.validate_ms as f32;
                                values[devtools_frame::SUBMIT as usize] = submitted_ms;
                                values[devtools_frame::HOUSEKEEPING as usize] = housekeeping_ms;
                                values[devtools_frame::TOTAL as usize] = (t.wait_ms + t.decode_ms + t.validate_ms) as f32 + submitted_ms + housekeeping_ms;
                                values[devtools_frame::PACKET as usize] = t.packet_bytes as f32;
                            }
                            Answer { values, ..Answer::ok(0.0) }
                        }
                        devtools_op::LOG => {
                            let mut values = Vec::new();
                            for (_id, source, failed, message) in logs.iter().filter(|(id,_,_,_)| *id >= request.a as u64) {
                                let text = pen_runtime::services::code_points(message);
                                values.extend([*source as f32, f32::from(u8::from(*failed)), text.len() as f32]);
                                values.extend(text);
                            }
                            Answer { values, ..Answer::ok(next_log as f32) }
                        }
                        devtools_op::COMMAND => { events.push(GameEvent::Console { command: request.a, argument: request.b }); Answer::ok(0.0) }
                        devtools_op::TUNE => { events.push(GameEvent::Tune { index: request.a as u32, value: request.b }); Answer::ok(0.0) }
                        devtools_op::PAUSE => { paused = true; Answer::ok(0.0) }
                        devtools_op::PLAY => { paused = false; Answer::ok(0.0) }
                        devtools_op::STEP => { stepping = (request.a.max(1.0) as u32).min(600); Answer::ok(0.0) }
                        devtools_op::REPLAY => {
                            position = position.saturating_sub((request.a.max(0.0) * 60.0) as usize);
                            replaying = !history.is_empty(); paused = true; Answer::ok(0.0)
                        }
                        devtools_op::HISTORY => { if history.is_empty() { history.push_back(game.moment().await?); } Answer::ok(0.0) }
                        devtools_op::REWIND => {
                            position = position.saturating_sub((request.a.max(0.0) * 60.0) as usize);
                            if let Some(moment) = history.get(position) { shown = game.restore(&mut screen, moment).await?; paused = true; events = preferences(); }
                            Answer::ok(0.0)
                        }
                        devtools_op::BAKE if development => {
                            let tuned = pen_runtime::tunables::tuned_values(&game.snapshot().await?.state);
                            match bake(&tuned).await { Ok(message) => Answer::text(&message), Err(error) => {
                                logs.push((next_log, 1, true, error.to_string())); next_log += 1; Answer::failed()
                            } }
                        }
                        devtools_op::TUNED => {
                            let mut tuned: Vec<_> = pen_runtime::tunables::tuned_values(&game.snapshot().await?.state).into_iter().collect();
                            tuned.sort_by_key(|(id,_)| *id);
                            Answer { values: tuned.into_iter().flat_map(|(id,value)| [id as f32,value]).collect(), ..Answer::ok(0.0) }
                        },
                        _ => Answer::unavailable(),
                    }
                } else { services::answer(request).await };
                tools_events.extend(answer.events(request.request));
            }
            if rendered.ui.iter().any(|node| node.focused) && !typing { events.push(GameEvent::FocusLost); }
            overlay = Some(rendered);
        }
        // UI opened by this frame also owns its queued keyboard/pointer input.
        if overlay.as_ref().is_some_and(|frame| frame.ui.iter().any(|node| node.focused)) {
            events.retain(|event| !matches!(event, GameEvent::KeyDown(_) | GameEvent::KeyUp(_)));
        }
        if over(overlay.as_ref(), pointer) {
            events.retain(|event| !matches!(event, GameEvent::PointerDown { .. } | GameEvent::PointerUp { .. } | GameEvent::PointerMove { .. } | GameEvent::Wheel { .. }));
        }
        if replaying {
            if let Some(moment) = history.get(position) {
                shown = game.restore(&mut screen, moment).await?;
                position += 1;
            }
            if position >= history.len() { position = history.len().saturating_sub(1); replaying = false; paused = false; events = preferences(); game.continue_clock(timestamp); }
        } else if !paused || stepping > 0 {
            stepping = stepping.saturating_sub(1);
            events.extend(screen.renderer.take_events());
            let input = game.input(&Frame { timestamp, viewport, events: std::mem::take(&mut events) })?;
            game.send(&input)?;
            shown = game.receive(&screen).await?;
            let housekeeping = web_time::Instant::now();
            game.follow(&mut screen, &shown).await?;
            housekeeping_ms = housekeeping.elapsed().as_secs_f32() * 1000.0;
            frame_number += 1;
            if !history.is_empty() {
                history.truncate(position + 1);
                history.push_back(game.moment().await?);
                if history.len() > 600 { history.pop_front(); }
                position = history.len() - 1;
            }
            for request in &shown.services { events.extend(services::answer(request).await.events(request.request)); }
            if let Some(values) = &shown.preferences && let Some(storage) = storage() { let _ = storage.set_item("pen.preferences", &serde_json::to_string(values)?); }
            if shown.save_requested { saves::write("pen.quicksave", &game.snapshot().await?).await?; }
            if shown.load_requested && let Some(saved) = saves::read("pen.quicksave").await { game.load(&mut screen, &saved).await?; events = preferences(); }
            if shown.scene_save_requested { saves::write("pen.scene", &game.scene().await?).await?; }
            if shown.scene_load_requested && let Some(saved) = saves::read("pen.scene").await { game.load_scene(&mut screen, &saved).await?; events = preferences(); }
            audio.submit(&shown, &catalog).await;
        }
        screen.canvas.element.set_attribute("data-frame", &frame_number.to_string()).map_err(error)?;
        screen.canvas.element.set_attribute("data-quality", &format!("msaa={} scale={}", shown.quality.msaa, shown.quality.scale)).map_err(error)?;
        screen.canvas.element.set_attribute("data-pending", &format!("game={} tools={}", screen.renderer.pending(), screen.overlay.pending())).map_err(error)?;
        let submitted = web_time::Instant::now();
        screen.draw(&shown, overlay.as_ref(), &catalog)?;
        submitted_ms = submitted.elapsed().as_secs_f32() * 1000.0;
        // Bound submitted frames on slow GPUs; RAF alone does not apply
        // backpressure to WebGPU and screenshots must not wait on a huge queue.
        screen.settled().await;
        let mut busy = "false";
        if !screen.renderer.quiet() || screen.renderer.pending() > 0 || screen.overlay.pending() > 0 { busy = "true"; }
        screen.canvas.element.set_attribute("aria-busy", busy).map_err(error)?;
        let ui = overlay.as_ref().filter(|f| !f.ui.is_empty()).map_or(shown.ui.as_slice(), |f| f.ui.as_slice());
        accessibility.update(ui)?;
        let mut cursor = "auto";
        if shown.hide_cursor || overlay.as_ref().is_some_and(|frame| frame.hide_cursor) { cursor = "none"; }
        screen.canvas.element.style().set_property("cursor", cursor).map_err(error)?;
        let status = window.document().unwrap().get_element_by_id("status");
        if let Some(status) = status { status.set_text_content(None); }
        if timestamp - last_save > 5.0 {
            saves::write("pen.autosave", &game.snapshot().await?).await?;
            if let Some(runtime) = tools.as_mut() { saves::write("pen.devtools", &runtime.snapshot().await?).await?; }
            last_save = timestamp;
        }
        if development && timestamp - last_reload > 1.0 {
            last_reload = timestamp;
            if let Ok(bytes) = fetch("/__pen/revision").await {
                let next = String::from_utf8_lossy(&bytes).into_owned();
                if !revision.is_empty() && next != revision {
                    let reloaded_catalog = fetch("catalog.json").await.ok()
                        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
                        .and_then(|json| pen_protocol::boundary::parse_catalog(&json).ok()).unwrap_or_else(|| catalog.clone());
                    if let Ok(module) = fetch("game.wasm").await {
                        let result = game.reload(&mut screen, module, reloaded_catalog.clone()).await;
                        if result == pen_runtime::game::ReloadResult::Activated { catalog = reloaded_catalog; }
                        logs.push((next_log, 2, result != pen_runtime::game::ReloadResult::Activated, format!("reload: {result:?}"))); next_log += 1;
                        events = preferences();
                    }
                    if let Ok(module) = fetch("devtools.wasm").await {
                        let shell = &mut Shell(viewport);
                        if let Some(runtime) = tools.as_mut() { runtime.reload(shell, module, catalog.clone()).await; }
                        else { tools = Some(Runtime::new(&BrowserBackend, module, catalog.clone(), shell, saves::read("pen.devtools").await, None).await?); }
                    }
                    screen.renderer.preload(&catalog);
                    screen.renderer.reload_all();
                    screen.overlay.reload_all();
                    audio.reload();
                }
                revision = next;
            }
        }
        if logs.len() > 400 { logs.drain(..logs.len() - 400); }
        if shown.exit_requested { break; }
    }
    saves::write("pen.autosave", &game.snapshot().await?).await?;
    game.close().await;
    if let Some(runtime) = tools.as_mut() { runtime.close().await; }
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
        if let Ok(scene) = snapshot::decode(&scene.to_vec()) && scene.schema == program.schema() {
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
        Ok::<_,anyhow::Error>(output)
    }.await;
    result.map_err(|error| JsValue::from_str(&error.to_string()))
}

/// Exercises real browser platform adapters without changing a game's saves.
#[wasm_bindgen]
pub async fn qualify_platform() -> Result<String, JsValue> {
    let result = async {
        let saved = SavedState { schema: 5, state: vec![1.0,-2.5,3.0] };
        saves::write("pen.qualification", &saved).await?;
        anyhow::ensure!(saves::read("pen.qualification").await == Some(saved), "browser snapshot round trip failed");
        let path = pen_protocol::ServiceRequest { request: 1, service: service::PATHFINDING, operation: pen_protocol::pathfinding_op::FIND,
            a: 3.0, b: 2.0, c: 0.0, d: 5.0, value_count: 6, values: vec![0.0,0.0,1.0,0.0,0.0,0.0] };
        let answer = services::answer(&path).await;
        anyhow::ensure!(answer.values == [0.0,1.0,4.0,5.0] && answer.value == 4.0, "browser pathfinding failed");
        let unavailable = services::answer(&pen_protocol::ServiceRequest { service: service::STEAM, ..pen_protocol::ServiceRequest::default() }).await;
        anyhow::ensure!(unavailable.status == pen_protocol::service_status::UNAVAILABLE, "browser Steam must be unavailable");
        let unknown = services::answer(&pen_protocol::ServiceRequest { service: 999, ..pen_protocol::ServiceRequest::default() }).await;
        anyhow::ensure!(unknown.status == pen_protocol::service_status::UNKNOWN, "unregistered browser service must be unknown");
        audio::qualify().await?;
        Ok::<_,anyhow::Error>("IndexedDB snapshot, pathfinding, service statuses and WebAudio graph passed".to_owned())
    }.await;
    result.map_err(|error| JsValue::from_str(&error.to_string()))
}
