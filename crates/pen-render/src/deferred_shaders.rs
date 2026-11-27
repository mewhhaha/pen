//! Lazy on-demand WGSL shader loader for browser builds (`feature = "deferred-shaders"`).
//!
//! Moves heavy optional pass shaders out of `pen_web_bg.wasm`'s Data section so
//! browsers only download and compile shaders for passes a game actually activates.
//! Native builds and builds without `deferred-shaders` embed shaders inline with zero
//! runtime overhead.

#[cfg(all(target_arch = "wasm32", feature = "deferred-shaders"))]
mod imp {
    use std::cell::RefCell;
    use std::collections::{HashMap, HashSet};

    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::{JsFuture, spawn_local};

    #[derive(Default)]
    struct Cache {
        loaded: HashMap<&'static str, &'static str>,
        in_flight: HashSet<&'static str>,
        failed: HashSet<&'static str>,
    }

    thread_local! {
        static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
    }

    async fn fetch_shader(name: &'static str) -> Result<String, String> {
        let url = format!("shaders/{name}");
        let global = js_sys::global();
        let promise = if let Some(window) = global.dyn_ref::<web_sys::Window>() {
            window.fetch_with_str(&url)
        } else {
            global.unchecked_ref::<web_sys::WorkerGlobalScope>().fetch_with_str(&url)
        };
        let response: web_sys::Response =
            JsFuture::from(promise).await.map_err(|e| format!("{url}: {e:?}"))?.unchecked_into();
        if !response.ok() {
            return Err(format!("{url}: HTTP {}", response.status()));
        }
        let buf = JsFuture::from(response.array_buffer().map_err(|e| format!("{url}: {e:?}"))?)
            .await
            .map_err(|e| format!("{url}: {e:?}"))?;
        let bytes = Uint8Array::new(&buf).to_vec();
        String::from_utf8(bytes).map_err(|e| format!("{url}: {e}"))
    }

    /// Returns the shader source if already fetched, or kicks off one background fetch
    /// and returns `None` until it arrives.
    pub(crate) fn get(name: &'static str) -> Option<&'static str> {
        CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if let Some(&src) = cache.loaded.get(name) {
                return Some(src);
            }
            if !cache.failed.contains(name) && cache.in_flight.insert(name) {
                spawn_local(async move {
                    let res = fetch_shader(name).await;
                    CACHE.with(|cell| {
                        let mut cache = cell.borrow_mut();
                        cache.in_flight.remove(name);
                        match res {
                            Ok(src) => {
                                let leaked: &'static str = Box::leak(src.into_boxed_str());
                                cache.loaded.insert(name, leaked);
                            }
                            Err(err) => {
                                log::warn!("deferred shader {name}: {err}");
                                cache.failed.insert(name);
                            }
                        }
                    });
                });
            }
            None
        })
    }

    /// Returns `true` once every named shader is cached, triggering background fetches
    /// for any that are still missing.
    pub(crate) fn ready_all(names: &[&'static str]) -> bool {
        let mut all = true;
        for &name in names {
            if get(name).is_none() {
                all = false;
            }
        }
        all
    }

    /// Async fetch for background pipeline jobs (`loader.rs`).
    #[cfg_attr(not(feature = "render-gpu-culling"), allow(dead_code))]
    pub(crate) async fn load_async(name: &'static str) -> Result<&'static str, String> {
        if let Some(src) = CACHE.with(|cell| cell.borrow().loaded.get(name).copied()) {
            return Ok(src);
        }
        let src = fetch_shader(name).await?;
        let leaked: &'static str = Box::leak(src.into_boxed_str());
        CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            cache.in_flight.remove(name);
            cache.loaded.insert(name, leaked);
        });
        Ok(leaked)
    }

    /// Returns a shader previously verified by `get` / `ready_all` / `load_async`.
    pub(crate) fn expect(name: &'static str) -> &'static str {
        CACHE.with(|cell| {
            cell.borrow()
                .loaded
                .get(name)
                .copied()
                .unwrap_or_else(|| panic!("deferred shader {name} used before ready"))
        })
    }

    /// Number of deferred shader fetches currently in flight.
    pub(crate) fn pending() -> usize {
        CACHE.with(|cell| cell.borrow().in_flight.len())
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "deferred-shaders")))]
#[allow(dead_code)]
mod imp {
    pub(crate) fn get(_name: &'static str) -> Option<&'static str> {
        Some("")
    }

    pub(crate) fn ready_all(_names: &[&'static str]) -> bool {
        true
    }

    pub(crate) fn pending() -> usize {
        0
    }
}

#[allow(unused_imports)]
pub(crate) use imp::*;
