// What differs between the native renderer and the one in the browser: where
// asset bytes come from (files under the asset root, or fetched from under
// its URL), how images are decoded, the clock, and how the loader's work runs
// (threads, or tasks on the page's event loop). Warnings and timings go to
// the `log` crate, for the host's logger.
use anyhow::Result;

use crate::texture::DecodedImage;

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::path::{Component, Path};

    use anyhow::{Result, bail};

    use crate::asset_store::validate_asset_key;
    use crate::texture::DecodedImage;

    /// The asset's bytes, from its file under `root`; keys cannot leave it.
    pub async fn read_asset(root: &str, key: &str) -> Result<Vec<u8>> {
        validate_asset_key(key)?;
        let relative = Path::new(key);
        if relative.components().any(|component| !matches!(component, Component::Normal(_))) {
            bail!("asset key must stay inside assets: {key}");
        }
        Ok(std::fs::read(Path::new(root).join(relative))?)
    }

    pub async fn decode_image(bytes: Vec<u8>) -> Result<DecodedImage> {
        crate::texture::decode_texture(&bytes)
    }

    /// Lets other work run between long steps; threads need not.
    pub async fn yield_now() {}

    /// Seconds on a monotonic clock, from when it was made.
    pub struct Clock(std::time::Instant);

    impl Clock {
        pub fn new() -> Self {
            Self(std::time::Instant::now())
        }
        pub fn now(&self) -> f64 {
            self.0.elapsed().as_secs_f64()
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use anyhow::{Result, anyhow, bail};
    use js_sys::{Array, Promise, Uint8Array};
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::JsFuture;

    use crate::asset_store::validate_asset_key;
    use crate::texture::DecodedImage;

    fn failed(error: JsValue) -> anyhow::Error {
        anyhow!(
            error
                .as_string()
                .or_else(|| js_sys::JSON::stringify(&error).ok().and_then(|text| text.as_string()))
                .unwrap_or_else(|| format!("{error:?}"))
        )
    }

    /// The page's global scope: a window, or a worker's.
    fn global() -> js_sys::Object {
        js_sys::global()
    }

    async fn fetch(url: &str) -> Result<web_sys::Response> {
        let global = global();
        let promise = if let Some(window) = global.dyn_ref::<web_sys::Window>() {
            window.fetch_with_str(url)
        } else {
            global.unchecked_ref::<web_sys::WorkerGlobalScope>().fetch_with_str(url)
        };
        let response: web_sys::Response = JsFuture::from(promise).await.map_err(failed)?.unchecked_into();
        if !response.ok() {
            let text = JsFuture::from(response.text().map_err(failed)?).await.map_err(failed)?;
            bail!("{url}: {} {}", response.status(), text.as_string().unwrap_or_default());
        }
        Ok(response)
    }

    /// The asset's bytes, fetched from under the asset root's URL.
    pub async fn read_asset(root: &str, key: &str) -> Result<Vec<u8>> {
        validate_asset_key(key)?;
        let url = format!("{}/{key}", root.trim_end_matches('/'));
        let response = fetch(&url).await?;
        let buffer = JsFuture::from(response.array_buffer().map_err(failed)?).await.map_err(failed)?;
        Ok(Uint8Array::new(&buffer).to_vec())
    }

    /// The browser's own decoder, off the page's thread: the image's bytes as
    /// they are, unconverted and not premultiplied.
    pub async fn decode_image(bytes: Vec<u8>) -> Result<DecodedImage> {
        let parts = Array::of1(&Uint8Array::from(bytes.as_slice()));
        let blob = web_sys::Blob::new_with_u8_array_sequence(&parts).map_err(failed)?;
        let options = web_sys::ImageBitmapOptions::new();
        options.set_color_space_conversion(web_sys::ColorSpaceConversion::None);
        options.set_premultiply_alpha(web_sys::PremultiplyAlpha::None);
        let global = global();
        let promise = if let Some(window) = global.dyn_ref::<web_sys::Window>() {
            window.create_image_bitmap_with_blob_and_image_bitmap_options(&blob, &options)
        } else {
            global
                .unchecked_ref::<web_sys::WorkerGlobalScope>()
                .create_image_bitmap_with_blob_and_image_bitmap_options(&blob, &options)
        }
        .map_err(failed)?;
        let bitmap: web_sys::ImageBitmap = JsFuture::from(promise).await.map_err(failed)?.unchecked_into();
        let (width, height) = (bitmap.width(), bitmap.height());
        let surface = web_sys::OffscreenCanvas::new(width, height).map_err(failed)?;
        let context: web_sys::OffscreenCanvasRenderingContext2d = surface
            .get_context("2d")
            .map_err(failed)?
            .ok_or_else(|| anyhow!("no 2d context to decode images with"))?
            .unchecked_into();
        context.draw_image_with_image_bitmap(&bitmap, 0.0, 0.0).map_err(failed)?;
        bitmap.close();
        let image = context.get_image_data(0.0, 0.0, f64::from(width), f64::from(height)).map_err(failed)?;
        Ok(DecodedImage { width, height, pixels: image.data().0 })
    }

    /// Lets the page draw a frame between long steps.
    pub async fn yield_now() {
        let promise = Promise::new(&mut |resolve, _| {
            let global = global();
            let set_timeout = js_sys::Reflect::get(&global, &"setTimeout".into()).unwrap();
            let _ = set_timeout.unchecked_ref::<js_sys::Function>().call2(&global, &resolve, &JsValue::from(0));
        });
        let _ = JsFuture::from(promise).await;
    }

    fn milliseconds() -> f64 {
        let performance = js_sys::Reflect::get(&global(), &"performance".into()).unwrap();
        performance.unchecked_into::<web_sys::Performance>().now()
    }

    /// Seconds on the page's monotonic clock, from when it was made.
    pub struct Clock(f64);

    impl Clock {
        pub fn new() -> Self {
            Self(milliseconds())
        }
        pub fn now(&self) -> f64 {
            (milliseconds() - self.0) / 1000.0
        }
    }
}

pub(crate) use imp::{Clock, yield_now};

pub(crate) async fn read_asset(root: &str, key: &str) -> Result<Vec<u8>> {
    imp::read_asset(root, key).await
}

pub(crate) async fn decode_image(bytes: Vec<u8>) -> Result<DecodedImage> {
    imp::decode_image(bytes).await
}
