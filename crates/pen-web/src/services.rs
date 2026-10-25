//! The game's services in a browser, through pen-runtime's registry: what
//! the game runs on, noise (`pen_simulation`,
//! the same bits as native), web addresses (a new tab, as the page's
//! permissions allow) and the clipboard, whose promises answer a later
//! frame. Steam and reports to the developer are unavailable.
use std::cell::RefCell;
use std::rc::Rc;

use pen_protocol::{clipboard_op, service, url_op};
use pen_runtime::services::{Answer, Platform, Request, Service, Services, Unavailable, openable};
use wasm_bindgen_futures::{JsFuture, spawn_local};

/// The browser's services.
pub fn standard() -> Services {
    let window = web_sys::window().unwrap();
    let mut services = Services::new();
    services.register(Box::new(Unavailable(service::STEAM)));
    services.register(Box::new(Platform {
        os: "web".into(),
        locale: window.navigator().language(),
        device: "browser".into(),
        handheld: false,
    }));
    services.register(Box::new(Url));
    services.register(Box::new(Clipboard::default()));
    pen_simulation::register(&mut services);
    // Reports go only from a native package whose developer takes them.
    services.register(Box::new(Unavailable(service::REPORTS)));
    services
}

/// The browser's services, and the engine's preferences as the page keeps
/// them.
pub fn with_preferences(preferences: pen_runtime::engine_preferences::PreferencesService) -> Services {
    let mut services = standard();
    services.register(Box::new(preferences));
    services
}

/// Opens a web address in a new tab.
struct Url;

impl Service for Url {
    fn code(&self) -> u32 {
        service::URL
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        if request.operation != url_op::OPEN {
            return Some(Answer::unknown());
        }
        let address = request.text();
        if !openable(&address) {
            return Some(Answer::failed());
        }
        let window = web_sys::window().unwrap();
        Some(match window.open_with_url_and_target_and_features(&address, "_blank", "noopener,noreferrer") {
            Ok(Some(_)) => Answer::ok(1.0),
            _ => Answer::unavailable(),
        })
    }
}

/// The page's clipboard, as its permissions allow: each request's promise
/// answers in a later frame.
#[derive(Default)]
struct Clipboard {
    answered: Rc<RefCell<Vec<(u32, Answer)>>>,
}

impl Service for Clipboard {
    fn code(&self) -> u32 {
        service::CLIPBOARD
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        let clipboard = web_sys::window().unwrap().navigator().clipboard();
        let (id, count) = (request.id, request.values.len());
        let promise = match request.operation {
            clipboard_op::COPY => clipboard.write_text(&request.text()),
            clipboard_op::PASTE => clipboard.read_text(),
            _ => return Some(Answer::unknown()),
        };
        let copying = request.operation == clipboard_op::COPY;
        let answered = self.answered.clone();
        spawn_local(async move {
            let answer = match JsFuture::from(promise).await {
                Ok(_) if copying => Answer::ok(count as f32),
                Ok(text) => Answer::text(&text.as_string().unwrap_or_default()),
                Err(_) => Answer::unavailable(),
            };
            answered.borrow_mut().push((id, answer));
        });
        None
    }

    fn poll(&mut self) -> Vec<(u32, Answer)> {
        std::mem::take(&mut *self.answered.borrow_mut())
    }
}
