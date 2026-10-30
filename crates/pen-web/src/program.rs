//! The browser's WebAssembly backend: a guest's long-lived `main(host)`
//! runs under JSPI (`guest.js`), suspended at each host call until the next
//! request. Rust checks the module's `blot:abi` manifest before it runs, and
//! describes a trap's Wasm frames with the module's symbols, as native pen
//! does.
use std::rc::Rc;

use anyhow::{Result, anyhow, ensure};
use js_sys::{Float32Array, Promise, Reflect, Uint8Array};
use pen_runtime::program::{self, Backend, Operation, Program};
use pen_runtime::symbols::{Symbols, js_frames};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_time::Instant;

#[wasm_bindgen(raw_module = "./guest.js")]
extern "C" {
    #[wasm_bindgen(catch, js_name = openGuest)]
    async fn open_guest(bytes: Uint8Array, ownership: bool) -> Result<JsValue, JsValue>;
    type Guest;
    #[wasm_bindgen(method, catch)]
    fn next(this: &Guest, values: &Float32Array) -> Result<Promise, JsValue>;
    #[wasm_bindgen(method, js_name = memoryBytes)]
    fn memory_bytes(this: &Guest) -> usize;
    #[wasm_bindgen(method)]
    fn close(this: &Guest) -> Promise;
}

/// A JavaScript exception as an error.
pub fn error(value: JsValue) -> anyhow::Error {
    anyhow!("{:?}", value)
}

/// A guest's failure as an error: the exception's message, and the Wasm
/// frames of its stack described by the module's symbols.
fn failure(value: JsValue, symbols: &Symbols) -> anyhow::Error {
    let stack = Reflect::get(&value, &"stack".into()).ok().and_then(|stack| stack.as_string());
    let frames = stack.as_deref().map(js_frames).unwrap_or_default();
    let error = match value.dyn_ref::<js_sys::Error>() {
        Some(exception) => anyhow!("{}", String::from(exception.to_string())),
        None => error(value),
    };
    if frames.is_empty() { error } else { error.context(symbols.backtrace(&frames)) }
}

#[derive(Clone)]
pub struct BrowserBackend;

pub struct BrowserProgram {
    guest: Guest,
    schema: u32,
    scenes: bool,
    published: bool,
    pending: Option<(Operation, Promise)>,
    stopping: bool,
    answered: Option<Instant>,
    failure: Option<String>,
    symbols: Rc<Symbols>,
}

impl Backend for BrowserBackend {
    type Module = Vec<u8>;
    type Program = BrowserProgram;

    async fn start(&self, module: &Vec<u8>) -> Result<BrowserProgram> {
        let abi = pen_runtime::abi::read_abi(module)?;
        // The mutable application exchange is serial. Isolated worker services
        // copy their own data and never share this invocation or its pools.
        ensure!(
            !abi.concurrent.iter().any(|name| name == "main"),
            "unsupported_abi: the game frame callback must be serial"
        );
        abi.check_game()?;
        let symbols = Rc::new(Symbols::read(module));
        let value = open_guest(Uint8Array::from(module.as_slice()), abi.ownership)
            .await
            .map_err(|value| failure(value, &symbols))?;
        let description = Float32Array::new(&Reflect::get(&value, &"description".into()).map_err(error)?).to_vec();
        let (schema, scenes) = program::handshake(&description)?;
        Ok(BrowserProgram {
            guest: value.unchecked_into(),
            schema,
            scenes,
            published: true,
            pending: None,
            stopping: false,
            answered: None,
            failure: None,
            symbols,
        })
    }
}

impl Program for BrowserProgram {
    fn schema(&self) -> u32 {
        self.schema
    }

    fn scenes(&self) -> bool {
        self.scenes
    }

    fn send(&mut self, operation: Operation, payload: &[f32]) -> Result<()> {
        ensure!(!self.stopping, "game program is closed");
        ensure!(self.pending.is_none(), "game program already has a request in flight");
        ensure!(self.failure.is_none(), "game program failed: {:?}", self.failure);
        let packet = program::request(self.published, operation, payload);
        let promise = self.guest.next(&Float32Array::from(packet.as_slice())).map_err(error)?;
        self.pending = Some((operation, promise));
        self.published = false;
        Ok(())
    }

    async fn response(&mut self) -> Result<program::Response> {
        let (operation, promise) =
            self.pending.take().ok_or_else(|| anyhow!("game program has no request in flight"))?;
        let result = JsFuture::from(promise).await.map_err(|value| failure(value, &self.symbols));
        self.answered = Some(Instant::now());
        match result {
            Ok(value) => program::response(operation, Float32Array::new(&value).to_vec()),
            Err(error) => {
                self.failure = Some(format!("{error:#}"));
                Err(error)
            }
        }
    }

    fn accept(&mut self) {
        self.published = true;
    }

    fn answered(&self) -> Option<Instant> {
        self.answered
    }

    fn memory_bytes(&self) -> usize {
        self.guest.memory_bytes()
    }

    async fn close(&mut self) {
        if self.stopping {
            return;
        }
        self.stopping = true;
        let _ = JsFuture::from(self.guest.close()).await;
    }
}
