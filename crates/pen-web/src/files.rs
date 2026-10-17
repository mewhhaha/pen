//! Files between the player and the page: a save or an input recording
//! handed over as a download, and those the player hands the page, picked
//! in the browser's file chooser or dropped onto it. What arrives, and the
//! word on what became of it, waits for the frame loop.
use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
use js_sys::{Array, Uint8Array};
use wasm_bindgen::{JsCast, closure::Closure};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{Blob, BlobPropertyBag, DragEvent, Event, EventTarget, File, HtmlAnchorElement, HtmlInputElement, Url};

use crate::program::error;

/// What the frame loop is told.
pub enum Arrival {
    /// A file the player handed the page: its name and bytes.
    File { name: String, bytes: Vec<u8> },
    /// A line for the log: whether an error, and the text.
    Note { error: bool, text: String },
}

#[derive(Clone, Default)]
pub struct Files {
    arrived: Rc<RefCell<Vec<Arrival>>>,
}

fn listen(target: &EventTarget, name: &str, call: impl FnMut(Event) + 'static) {
    let callback = Closure::<dyn FnMut(Event)>::new(call);
    target.add_event_listener_with_callback(name, callback.as_ref().unchecked_ref()).unwrap();
    callback.forget();
}

impl Files {
    /// Takes files dropped onto the page.
    pub fn install() -> Files {
        let files = Files::default();
        let window = web_sys::window().unwrap();
        listen(&window, "dragover", |event| event.prevent_default());
        let dropped = files.clone();
        listen(&window, "drop", move |event| {
            event.prevent_default();
            let Some(list) = event.unchecked_ref::<DragEvent>().data_transfer().and_then(|transfer| transfer.files())
            else {
                return;
            };
            for index in 0..list.length() {
                if let Some(file) = list.item(index) {
                    dropped.read(file);
                }
            }
        });
        files
    }

    /// Says `text` in the log with the next frame.
    pub fn note(&self, error: bool, text: impl Into<String>) {
        self.arrived.borrow_mut().push(Arrival::Note { error, text: text.into() });
    }

    /// What arrived since the frame loop last looked.
    pub fn take(&self) -> Vec<Arrival> {
        std::mem::take(&mut *self.arrived.borrow_mut())
    }

    /// Reads `file` for the frame loop.
    fn read(&self, file: File) {
        let files = self.clone();
        spawn_local(async move {
            let name = file.name();
            match JsFuture::from(file.array_buffer()).await {
                Ok(buffer) => {
                    let bytes = Uint8Array::new(&buffer).to_vec();
                    files.arrived.borrow_mut().push(Arrival::File { name, bytes });
                }
                Err(failure) => files.note(true, format!("reading {name}: {:#}", error(failure))),
            }
        });
    }

    /// Asks the player for a file (`accept` lists the extensions offered),
    /// which arrives once chosen. Browsers open their chooser only soon
    /// after the player pressed a key or clicked.
    pub fn choose(&self, accept: &str) -> Result<()> {
        let document = web_sys::window().unwrap().document().unwrap();
        let input: HtmlInputElement = document.create_element("input").map_err(error)?.unchecked_into();
        input.set_type("file");
        input.set_accept(accept);
        let files = self.clone();
        let chosen = input.clone();
        listen(&input, "change", move |_| {
            if let Some(file) = chosen.files().and_then(|list| list.item(0)) {
                files.read(file);
            }
        });
        input.click();
        Ok(())
    }
}

/// Hands `bytes` to the player as a download named `name`.
pub fn download(name: &str, bytes: &[u8], kind: &str) -> Result<()> {
    let parts = Array::of1(&Uint8Array::from(bytes));
    let options = BlobPropertyBag::new();
    options.set_type(kind);
    let blob = Blob::new_with_u8_array_sequence_and_options(&parts, &options).map_err(error)?;
    let url = Url::create_object_url_with_blob(&blob).map_err(error)?;
    let document = web_sys::window().unwrap().document().unwrap();
    let anchor: HtmlAnchorElement = document.create_element("a").map_err(error)?.unchecked_into();
    anchor.set_href(&url);
    anchor.set_download(name);
    anchor.click();
    // Once the browser has taken it.
    let revoke = Closure::once_into_js(move || {
        let _ = Url::revoke_object_url(&url);
    });
    let _ = web_sys::window()
        .unwrap()
        .set_timeout_with_callback_and_timeout_and_arguments_0(revoke.unchecked_ref(), 60_000);
    Ok(())
}

/// A name for a file made now: `PREFIX-20261016-142301.EXTENSION` (UTC),
/// as native pen names its shots and recordings.
pub fn named(prefix: &str, extension: &str) -> String {
    let now = js_sys::Date::new_0();
    format!(
        "{prefix}-{:04}{:02}{:02}-{:02}{:02}{:02}.{extension}",
        now.get_utc_full_year(),
        now.get_utc_month() + 1,
        now.get_utc_date(),
        now.get_utc_hours(),
        now.get_utc_minutes(),
        now.get_utc_seconds()
    )
}

/// The last part of `path`, as a download's name.
pub fn base_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}
