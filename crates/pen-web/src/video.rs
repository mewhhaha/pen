//! Browser video texture playback registry for `frame.video_streams`.
//! Lazily fetches `.y4m` streams over HTTP from the asset root, decodes frames
//! via `pen_render::VideoPlayer`, and uploads changed frames into `Renderer`.
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use anyhow::Result;
use pen_protocol::{Catalog, RenderFrame};
use pen_render::{Renderer, VideoPlayer, Y4mStream};
use wasm_bindgen_futures::spawn_local;

struct ActiveVideo {
    player: VideoPlayer,
    fps: f64,
    last_frame_index: Option<usize>,
}

enum Slot {
    Loading,
    Ready(ActiveVideo),
    Failed,
}

/// Manages asynchronous asset fetching and per-frame GPU uploads for `frame.video_streams` in `pen-web`.
pub struct WebVideoRegistry {
    assets: String,
    /// Map from catalog material key -> resolved video texture key (`None` while loading or if not video).
    material_keys: Rc<RefCell<HashMap<String, Option<String>>>>,
    /// Map from video texture key -> decoded `VideoPlayer` state.
    streams: Rc<RefCell<HashMap<String, Slot>>>,
}

impl WebVideoRegistry {
    pub fn new(assets: String) -> Self {
        Self {
            assets,
            material_keys: Rc::new(RefCell::new(HashMap::new())),
            streams: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    /// Steps all active `frame.video_streams`, kicking off lazy HTTP fetches for unseen
    /// materials/streams and uploading changed video frames into `renderer`.
    pub fn update(&mut self, frame: &RenderFrame, catalog: &Catalog, renderer: &mut Renderer) -> Result<()> {
        for spec in &frame.video_streams {
            let Some(mat_key) = catalog.materials.get(spec.material as usize) else {
                continue;
            };
            let resolved = {
                let mut map = self.material_keys.borrow_mut();
                if let Some(existing) = map.get(mat_key) {
                    existing.clone()
                } else {
                    map.insert(mat_key.clone(), None);
                    let mat_key_owned = mat_key.clone();
                    let url = format!("{}/{}", self.assets.trim_end_matches('/'), mat_key);
                    let material_keys = self.material_keys.clone();
                    let streams = self.streams.clone();
                    let assets = self.assets.clone();
                    let looping = spec.looping;
                    spawn_local(async move {
                        let Ok(mat_bytes) = crate::fetch(&url).await else { return };
                        let Ok((_, textures)) = pen_render::inspect_material(&mat_bytes) else { return };
                        let Some(tex_key) = textures.first().cloned() else { return };
                        material_keys.borrow_mut().insert(mat_key_owned, Some(tex_key.clone()));
                        if streams.borrow().contains_key(&tex_key) {
                            return;
                        }
                        streams.borrow_mut().insert(tex_key.clone(), Slot::Loading);
                        let video_url = format!("{}/{}", assets.trim_end_matches('/'), tex_key);
                        let slot = match crate::fetch(&video_url).await {
                            Ok(bytes) if bytes.starts_with(b"YUV4MPEG2") => match Y4mStream::parse(&bytes)
                                .and_then(|stream| {
                                    let fps = stream.fps();
                                    VideoPlayer::from_y4m(stream, looping).map(|player| (player, fps))
                                }) {
                                Ok((player, fps)) => {
                                    Slot::Ready(ActiveVideo { player, fps, last_frame_index: None })
                                }
                                Err(_) => Slot::Failed,
                            },
                            _ => Slot::Failed,
                        };
                        streams.borrow_mut().insert(tex_key, slot);
                    });
                    None
                }
            };
            let Some(tex_key) = resolved else { continue };
            let mut streams = self.streams.borrow_mut();
            let Some(Slot::Ready(active)) = streams.get_mut(&tex_key) else {
                continue;
            };
            let speed = if spec.speed <= 0.0 { 1.0 } else { spec.speed };
            active.player.looping = spec.looping;
            active.player.paused = !spec.playing;
            if spec.playing {
                let target_t = f64::from(spec.time.max(0.0)) * f64::from(speed);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "non-negative frame index"
                )]
                let target_idx = (target_t * active.fps).floor().max(0.0) as usize;
                active.player.seek_frame(target_idx);
            }
            let idx = active.player.current_index();
            if active.last_frame_index != Some(idx) {
                renderer.upload_video_frame(&tex_key, active.player.current_frame())?;
                active.last_frame_index = Some(idx);
            }
        }
        Ok(())
    }
}
