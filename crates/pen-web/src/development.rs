//! `pen web`'s development endpoints, off the frame loop: new builds as the
//! served revision moves on, and baking tunables into the checkout. Both
//! run as tasks of their own; the frame loop takes what they bring when the
//! game is idle.
use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
use js_sys::Promise;
use pen_protocol::Catalog;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::Response;

use crate::fetch;
use crate::program::error;

/// How often the served revision is asked for.
const POLL_MS: i32 = 1000;

/// A new build: what of it fetched.
pub struct Update {
    pub catalog: Option<Catalog>,
    pub game: Option<Vec<u8>>,
    pub devtools: Option<Vec<u8>>,
}

/// The endpoints' word for the frame loop.
#[derive(Default)]
struct Mail {
    update: Option<Update>,
    /// Log lines: whether an error, and the text.
    lines: Vec<(bool, String)>,
    /// A bake is under way.
    baking: bool,
}

pub struct Development {
    mail: Rc<RefCell<Mail>>,
}

/// Resolves after `ms`.
async fn sleep(ms: i32) {
    let promise = Promise::new(&mut |resolve, _| {
        let _ = web_sys::window().unwrap().set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
    });
    let _ = JsFuture::from(promise).await;
}

async fn catalog() -> Result<Catalog> {
    let json: serde_json::Value = serde_json::from_slice(&fetch("catalog.json").await?)?;
    pen_protocol::boundary::parse_catalog(&json)
}

/// Asks for the revision now and then; when it moves on, fetches the new
/// build for the frame loop (a newer one replaces one not yet taken).
async fn watch(mail: Rc<RefCell<Mail>>) {
    let mut revision: Option<String> = None;
    loop {
        sleep(POLL_MS).await;
        let Ok(bytes) = fetch("/__pen/revision").await else { continue };
        let next = String::from_utf8_lossy(&bytes).into_owned();
        if revision.as_ref().is_some_and(|revision| *revision != next) {
            let update = Update {
                catalog: catalog().await.ok(),
                game: fetch("game.wasm").await.ok(),
                devtools: fetch("devtools.wasm").await.ok(),
            };
            mail.borrow_mut().update = Some(update);
        }
        revision = Some(next);
    }
}

/// POSTs `[id, value]` pairs to the bake endpoint; its text says what it did.
async fn post(values: &[(u32, f32)]) -> Result<String> {
    let options = web_sys::RequestInit::new();
    options.set_method("POST");
    options.set_body(&JsValue::from_str(&serde_json::to_string(values)?));
    let request = web_sys::Request::new_with_str_and_init("/__pen/bake", &options).map_err(error)?;
    request.headers().set("Content-Type", "application/json").map_err(error)?;
    let response: Response = JsFuture::from(web_sys::window().unwrap().fetch_with_request(&request))
        .await
        .map_err(error)?
        .dyn_into()
        .map_err(error)?;
    let message = JsFuture::from(response.text().map_err(error)?).await.map_err(error)?.as_string().unwrap_or_default();
    anyhow::ensure!(response.ok(), "{message}");
    Ok(message)
}

impl Development {
    /// Starts watching the revision.
    pub fn watch() -> Development {
        let mail = Rc::new(RefCell::new(Mail::default()));
        spawn_local(watch(mail.clone()));
        Development { mail }
    }

    /// A new build fetched since the last call.
    pub fn take_update(&self) -> Option<Update> {
        self.mail.borrow_mut().update.take()
    }

    /// What the endpoints said since the last call, for the log.
    pub fn take_lines(&self) -> Vec<(bool, String)> {
        std::mem::take(&mut self.mail.borrow_mut().lines)
    }

    /// Bakes the tuned values, sorted by id, into the checkout; the log
    /// says how it went. False while another bake is under way.
    pub fn bake(&self, mut values: Vec<(u32, f32)>) -> bool {
        if std::mem::replace(&mut self.mail.borrow_mut().baking, true) {
            return false;
        }
        values.sort_by_key(|(id, _)| *id);
        let mail = self.mail.clone();
        spawn_local(async move {
            let line = match post(&values).await {
                Ok(said) => (false, format!("bake: {said}")),
                Err(error) => (true, format!("bake: {error:#}")),
            };
            let mut mail = mail.borrow_mut();
            mail.lines.push(line);
            mail.baking = false;
        });
        true
    }
}
