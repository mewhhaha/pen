//! Native snapshot bytes in IndexedDB: saves can exceed localStorage's quota.
use crate::program::error;
use anyhow::Result;
use js_sys::{Promise, Uint8Array};
use pen_runtime::snapshot::{self, SavedState};
use wasm_bindgen::{JsCast, closure::Closure};
use wasm_bindgen_futures::JsFuture;
use web_sys::{IdbDatabase, IdbRequest, IdbTransaction, IdbTransactionMode};

async fn request(request: &IdbRequest) -> Result<wasm_bindgen::JsValue> {
    let mut success = None;
    let mut failure = None;
    let promise = Promise::new(&mut |resolve, reject| {
        let result = request.clone();
        let reject_result = reject.clone();
        success = Some(Closure::once(move |_: web_sys::Event| match result.result() {
            Ok(value) => {
                let _ = resolve.call1(&wasm_bindgen::JsValue::NULL, &value);
            }
            Err(value) => {
                let _ = reject_result.call1(&wasm_bindgen::JsValue::NULL, &value);
            }
        }));
        let failed = request.clone();
        failure = Some(Closure::once(move |_: web_sys::Event| {
            let message = failed
                .error()
                .ok()
                .flatten()
                .map(|e| e.message())
                .unwrap_or_else(|| "save database request failed".into());
            let _ = reject.call1(&wasm_bindgen::JsValue::NULL, &message.into());
        }));
        request.set_onsuccess(Some(success.as_ref().unwrap().as_ref().unchecked_ref()));
        request.set_onerror(Some(failure.as_ref().unwrap().as_ref().unchecked_ref()));
    });
    let result = JsFuture::from(promise).await.map_err(error);
    request.set_onsuccess(None);
    request.set_onerror(None);
    result
}
async fn committed(transaction: &IdbTransaction) -> Result<()> {
    let mut success = None;
    let mut failure = None;
    let promise = Promise::new(&mut |resolve, reject| {
        success = Some(Closure::once(move |_: web_sys::Event| {
            let _ = resolve.call0(&wasm_bindgen::JsValue::NULL);
        }));
        failure = Some(Closure::once(move |_: web_sys::Event| {
            let _ = reject.call1(&wasm_bindgen::JsValue::NULL, &"save transaction failed".into());
        }));
        transaction.set_oncomplete(Some(success.as_ref().unwrap().as_ref().unchecked_ref()));
        transaction.set_onabort(Some(failure.as_ref().unwrap().as_ref().unchecked_ref()));
    });
    let result = JsFuture::from(promise).await.map_err(error);
    transaction.set_oncomplete(None);
    transaction.set_onabort(None);
    result.map(|_| ())
}
async fn open() -> Result<IdbDatabase> {
    let factory = web_sys::window()
        .unwrap()
        .indexed_db()
        .map_err(error)?
        .ok_or_else(|| anyhow::anyhow!("browser saves are unavailable"))?;
    let opening = factory.open_with_u32("pen", 1).map_err(error)?;
    let upgrade = opening.clone();
    let callback = Closure::once(move |_: web_sys::Event| {
        if let Ok(value) = upgrade.result()
            && let Ok(database) = value.dyn_into::<IdbDatabase>()
        {
            let _ = database.create_object_store("saves");
        }
    });
    opening.set_onupgradeneeded(Some(callback.as_ref().unchecked_ref()));
    let result = request(&opening).await;
    opening.set_onupgradeneeded(None);
    result?.dyn_into().map_err(error)
}
fn key(name: &str) -> String {
    let path = web_sys::window().unwrap().location().pathname().unwrap_or_default();
    format!("{path}:{name}")
}
pub async fn read(name: &str) -> Option<SavedState> {
    let result = async {
        let database = open().await?;
        let transaction =
            database.transaction_with_str_and_mode("saves", IdbTransactionMode::Readonly).map_err(error)?;
        let store = transaction.object_store("saves").map_err(error)?;
        let value = request(&store.get(&key(name).into()).map_err(error)?).await?;
        database.close();
        if value.is_undefined() {
            return Ok(None);
        }
        snapshot::decode(&Uint8Array::new(&value).to_vec()).map(Some)
    }
    .await;
    match result {
        Ok(saved) => saved,
        Err(error) => {
            web_sys::console::warn_1(&format!("reading save: {error}").into());
            None
        }
    }
}
pub async fn write(name: &str, saved: &SavedState) -> Result<()> {
    let bytes = snapshot::encode(saved)?;
    let database = open().await?;
    let transaction = database.transaction_with_str_and_mode("saves", IdbTransactionMode::Readwrite).map_err(error)?;

    let store = transaction.object_store("saves").map_err(error)?;
    store.put_with_key(&Uint8Array::from(bytes.as_slice()), &key(name).into()).map_err(error)?;
    let result = committed(&transaction).await;
    database.close();
    result
}
