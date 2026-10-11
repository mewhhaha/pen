use anyhow::{Result, anyhow, ensure};
use js_sys::{Float32Array, Promise, Reflect, Uint8Array};
use pen_runtime::program::{self, Backend, Operation, Program};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_time::Instant;

#[wasm_bindgen(raw_module = "./guest.js")]
extern "C" {
    #[wasm_bindgen(catch, js_name = openGuest)]
    async fn open_guest(bytes: Uint8Array) -> Result<JsValue, JsValue>;
    type Guest;
    #[wasm_bindgen(method, catch)]
    fn next(this: &Guest, values: &Float32Array) -> Result<Promise, JsValue>;
    #[wasm_bindgen(method, js_name = memoryBytes)]
    fn memory_bytes(this: &Guest) -> usize;
    #[wasm_bindgen(method)]
    fn close(this: &Guest) -> Promise;
}
pub fn error(value: JsValue) -> anyhow::Error { anyhow!("{:?}", value) }

#[derive(Clone)]
pub struct BrowserBackend;
pub struct BrowserProgram {
    guest: Guest,
    schema: u32,
    scenes: bool,
    commit: bool,
    pending: Option<(Operation, Promise)>,
    stopping: bool,
    answered: Option<Instant>,
    failure: Option<String>,
}
impl Backend for BrowserBackend {
    type Module = Vec<u8>;
    type Program = BrowserProgram;
    async fn start(&self, module: &Vec<u8>) -> Result<BrowserProgram> {
        let value = open_guest(Uint8Array::from(module.as_slice())).await.map_err(error)?;
        let description = Float32Array::new(&Reflect::get(&value, &"description".into()).map_err(error)?).to_vec();
        let (schema, scenes) = program::handshake(&description)?;
        Ok(BrowserProgram { guest: value.unchecked_into(), schema, scenes, commit: true,
            pending: None, stopping: false, answered: None, failure: None })
    }
}
impl Program for BrowserProgram {
    fn schema(&self) -> u32 { self.schema }
    fn scenes(&self) -> bool { self.scenes }
    fn send(&mut self, operation: Operation, payload: &[f32]) -> Result<()> {
        ensure!(!self.stopping, "game program is closed");
        ensure!(self.pending.is_none(), "game program already has a request in flight");
        ensure!(self.failure.is_none(), "game program failed: {:?}", self.failure);
        let packet = program::request(self.commit, operation, payload);
        let promise = self.guest.next(&Float32Array::from(packet.as_slice())).map_err(error)?;
        self.pending = Some((operation, promise));
        self.commit = false;
        Ok(())
    }
    async fn response(&mut self) -> Result<Vec<f32>> {
        let (operation, promise) = self.pending.take().ok_or_else(|| anyhow!("game program has no request in flight"))?;
        let result = JsFuture::from(promise).await.map_err(error);
        self.answered = Some(Instant::now());
        match result {
            Ok(value) => program::response(operation, Float32Array::new(&value).to_vec()),
            Err(error) => { self.failure = Some(error.to_string()); Err(error) }
        }
    }
    fn accept(&mut self) { self.commit = true; }
    fn answered(&self) -> Option<Instant> { self.answered }
    fn memory_bytes(&self) -> usize { self.guest.memory_bytes() }
    async fn close(&mut self) {
        if self.stopping { return; }
        self.stopping = true;
        let _ = JsFuture::from(self.guest.close()).await;
    }
}
