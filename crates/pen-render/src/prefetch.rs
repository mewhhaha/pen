// Assets read and decoded before there is a renderer to load them: a host
// that knows what its scene draws first (pen's scenes.json) starts this at
// once, so the files are read, the meshes parsed and the images and packed
// textures decoded on threads of their own while the GPU opens and the
// program starts. The renderer it is given (`Renderer::warm`) takes each
// asset's first load from here instead of reading the file again, waiting
// for one still being decoded. In the browser it does nothing: the page
// fetches what it loads.
use std::collections::HashMap;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::asset_store::{AssetKind, Material, Mesh};
use crate::packed_texture::{PackedHeader, PackedLevel};
use crate::texture::DecodedImage;

/// An asset made ready ahead of its load.
#[cfg_attr(target_arch = "wasm32", allow(dead_code, reason = "the browser reads nothing ahead"))]
pub(crate) enum Prepared {
    // Boxed: a mesh carries its bounds, levels of detail and skin.
    Mesh(Box<Mesh>),
    Material(Material),
    Image(DecodedImage),
    /// A packed texture with every level inflated.
    Packed(PackedHeader, Vec<PackedLevel>),
    /// What needs the GPU to go further (a shader), read.
    Bytes(Vec<u8>),
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code, reason = "the browser reads nothing ahead"))]
enum State {
    Waiting,
    Working,
    Done(Option<Prepared>),
}

struct Shared {
    states: Mutex<HashMap<(AssetKind, String), State>>,
    done: Condvar,
}

/// Assets being read and decoded ahead of a renderer.
pub struct Prefetch {
    shared: Arc<Shared>,
}

impl Prefetch {
    /// Reads and decodes the assets under `root`, `(kind, key)` pairs with
    /// the kinds as `preload_scene` names them, in their order, on a few
    /// threads of its own.
    pub fn start(root: PathBuf, assets: &[(String, String)]) -> Prefetch {
        let order: Vec<(AssetKind, String)> =
            assets.iter().filter_map(|(kind, key)| Some((AssetKind::named(kind)?, key.clone()))).collect();
        let states = order.iter().map(|asset| (asset.clone(), State::Waiting)).collect();
        let shared = Arc::new(Shared { states: Mutex::new(states), done: Condvar::new() });
        #[cfg(not(target_arch = "wasm32"))]
        {
            let order = Arc::new(order);
            let next = Arc::new(AtomicUsize::new(0));
            let root: Arc<str> = root.to_string_lossy().into();
            // Two: the program loading and the GPU opening run beside them.
            // (At idle priority they were starved holding locks the GPU's
            // and the program's start wait on, which then took seconds.)
            for index in 0..2.min(order.len()) {
                let (shared, order, next, root) = (shared.clone(), order.clone(), next.clone(), root.clone());
                let _ = std::thread::Builder::new().name(format!("pen-prefetch-{index}")).spawn(move || {
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        let Some(asset) = order.get(at) else { return };
                        {
                            let mut states = shared.states.lock().unwrap();
                            match states.get_mut(asset) {
                                Some(state @ State::Waiting) => *state = State::Working,
                                // Taken before its turn: the loader reads it.
                                _ => continue,
                            }
                        }
                        let prepared = prepare(&root, asset.0, &asset.1);
                        let mut states = shared.states.lock().unwrap();
                        if let Some(state) = states.get_mut(asset) {
                            *state = State::Done(prepared);
                        }
                        shared.done.notify_all();
                    }
                });
            }
        }
        #[cfg(target_arch = "wasm32")]
        let _ = root;
        Prefetch { shared }
    }

    /// The asset made ready, waiting for it if it is being made; None when
    /// it was not asked for, has not begun, or could not be made (its load
    /// then reads it and says why).
    pub(crate) fn take(&self, kind: AssetKind, key: &str) -> Option<Prepared> {
        let asset = (kind, key.to_owned());
        let mut states = self.shared.states.lock().unwrap();
        loop {
            match states.remove(&asset)? {
                State::Done(prepared) => return prepared,
                State::Waiting => return None,
                State::Working => {
                    states.insert(asset.clone(), State::Working);
                    states = self.shared.done.wait(states).unwrap();
                }
            }
        }
    }
}

/// Reads and decodes one asset as the loader would, short of the GPU.
#[cfg(not(target_arch = "wasm32"))]
fn prepare(root: &str, kind: AssetKind, key: &str) -> Option<Prepared> {
    use crate::asset_store::{parse_material, parse_mesh};
    use crate::packed_texture::{PackedReader, is_packed_texture};
    let bytes = pollster::block_on(crate::platform::read_asset(root, key)).ok()?;
    match kind {
        AssetKind::Mesh => parse_mesh(&bytes).ok().map(|mesh| Prepared::Mesh(Box::new(mesh))),
        AssetKind::Material => parse_material(&bytes).ok().map(Prepared::Material),
        AssetKind::Texture if is_packed_texture(&bytes) => {
            let mut reader = PackedReader::open(bytes).ok()?;
            let mut levels = Vec::new();
            while let Some(level) = reader.next_level().ok()? {
                levels.push(level);
            }
            Some(Prepared::Packed(reader.header, levels))
        }
        AssetKind::Texture => pollster::block_on(crate::platform::decode_image(bytes)).ok().map(Prepared::Image),
        AssetKind::Shader | AssetKind::SpriteShader => Some(Prepared::Bytes(bytes)),
    }
}
