//! Browser video texture playback registry for `frame.video_streams`.
//! Lazily fetches `.y4m` streams over HTTP from the asset root with concurrency,
//! byte-budget, dimension, and key validation, indexes frames in a single backing
//! store via `pen_render::IndexedY4mStream`, uploads changed frames zero-copy
//! into `Renderer`, and evicts idle CPU/GPU streams.
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use pen_protocol::{Catalog, RenderFrame};
use pen_render::{ColorMatrix, IndexedY4mStream, Renderer, VideoPlayer};
use wasm_bindgen_futures::spawn_local;

/// Maximum simultaneous in-flight HTTP video fetches.
const MAX_CONCURRENT_FETCHES: usize = 2;
/// Maximum payload bytes accepted for a single video stream (32 MiB).
const MAX_STREAM_BYTES: usize = 32 * 1024 * 1024;
/// Cumulative byte budget across all resident video streams (64 MiB).
const MAX_TOTAL_BYTES: usize = 64 * 1024 * 1024;
/// Evict resident CPU + GPU video entries after this many unused frames.
const EVICT_IDLE_FRAMES: u64 = 180;
/// Base retry backoff after transient fetch/parse errors (frames).
const RETRY_BASE_FRAMES: u64 = 60;
const RETRY_MAX_FRAMES: u64 = 600;

struct ActiveVideo {
    player: VideoPlayer,
    last_frame_index: Option<usize>,
    last_seek: Option<(u32, u32, bool)>,
    last_used_frame: u64,
}

enum MaterialSlot {
    Loading,
    Resolved(Option<String>),
    Failed { retry_at: u64, attempts: u32 },
}

enum StreamSlot {
    Loading,
    Ready(ActiveVideo),
    Failed { retry_at: u64, attempts: u32 },
}

struct SharedState {
    in_flight: usize,
    total_bytes: usize,
    material_keys: HashMap<String, MaterialSlot>,
    streams: HashMap<String, StreamSlot>,
    errors: Vec<String>,
}

/// Manages bounded, validated asynchronous asset fetching and per-frame GPU uploads
/// for `frame.video_streams` in `pen-web`.
pub struct WebVideoRegistry {
    assets: String,
    frame_counter: u64,
    shared: Rc<RefCell<SharedState>>,
}

impl WebVideoRegistry {
    pub fn new(assets: String) -> Self {
        Self {
            assets,
            frame_counter: 0,
            shared: Rc::new(RefCell::new(SharedState {
                in_flight: 0,
                total_bytes: 0,
                material_keys: HashMap::new(),
                streams: HashMap::new(),
                errors: Vec::new(),
            })),
        }
    }

    /// Steps all active `frame.video_streams`, kicking off bounded lazy HTTP fetches for unseen
    /// materials/streams, uploading changed video frames into `renderer`, evicting idle streams,
    /// and returning any per-stream error diagnostics without aborting sibling streams.
    pub fn update(&mut self, frame: &RenderFrame, catalog: &Catalog, renderer: &mut Renderer) -> Vec<String> {
        self.frame_counter = self.frame_counter.saturating_add(1);
        let now_frame = self.frame_counter;
        let max_dim = renderer.max_texture_dimension_2d();
        let mut diagnostics = {
            let mut st = self.shared.borrow_mut();
            std::mem::take(&mut st.errors)
        };

        // Deduplicate directives by resolved texture key (last directive wins per key) so a shared
        // texture key never uploads conflicting frames within a single tick.
        let mut directives: Vec<(String, &pen_protocol::VideoStreamSpec)> = Vec::new();
        let mut seen_keys: HashMap<String, usize> = HashMap::new();
        let mut touched_keys: HashSet<String> = HashSet::new();

        for spec in &frame.video_streams {
            let Some(mat_key) = catalog.materials.get(spec.material as usize) else {
                diagnostics.push(format!("video stream material index {} out of catalog bounds", spec.material));
                continue;
            };
            if let Err(err) = pen_render::validate_asset_key(mat_key) {
                diagnostics.push(format!("video stream material {mat_key:?}: {err:#}"));
                continue;
            }
            let resolved = {
                let mut st = self.shared.borrow_mut();
                match st.material_keys.get(mat_key) {
                    Some(MaterialSlot::Resolved(tex)) => tex.clone(),
                    Some(MaterialSlot::Loading) => None,
                    Some(MaterialSlot::Failed { retry_at, .. }) if now_frame < *retry_at => None,
                    other => {
                        let attempts = match other {
                            Some(MaterialSlot::Failed { attempts, .. }) => *attempts,
                            _ => 0,
                        };
                        if st.in_flight >= MAX_CONCURRENT_FETCHES {
                            None
                        } else {
                            st.in_flight += 1;
                            st.material_keys.insert(mat_key.clone(), MaterialSlot::Loading);
                            let mat_key_owned = mat_key.clone();
                            let url = format!("{}/{}", self.assets.trim_end_matches('/'), mat_key);
                            let shared = self.shared.clone();
                            let assets = self.assets.clone();
                            let looping = spec.looping;
                            spawn_local(async move {
                                let mat_res = crate::fetch_bounded(&url, 1024 * 1024)
                                    .await
                                    .and_then(|bytes| pen_render::inspect_material_base_texture(&bytes));
                                let tex_key = match mat_res {
                                    Ok(Some(tex_key)) => {
                                        if let Err(err) = pen_render::validate_asset_key(&tex_key) {
                                            let mut st = shared.borrow_mut();
                                            st.in_flight = st.in_flight.saturating_sub(1);
                                            let next_attempts = attempts.saturating_add(1);
                                            let backoff = (RETRY_BASE_FRAMES << next_attempts.min(4)).min(RETRY_MAX_FRAMES);
                                            st.material_keys.insert(
                                                mat_key_owned.clone(),
                                                MaterialSlot::Failed { retry_at: now_frame + backoff, attempts: next_attempts },
                                            );
                                            st.errors.push(format!("video material {mat_key_owned}: invalid texture key {tex_key:?}: {err:#}"));
                                            return;
                                        }
                                        let mut st = shared.borrow_mut();
                                        st.material_keys
                                            .insert(mat_key_owned.clone(), MaterialSlot::Resolved(Some(tex_key.clone())));
                                        tex_key
                                    }
                                    Ok(None) => {
                                        let mut st = shared.borrow_mut();
                                        st.in_flight = st.in_flight.saturating_sub(1);
                                        st.material_keys.insert(mat_key_owned, MaterialSlot::Resolved(None));
                                        return;
                                    }
                                    Err(err) => {
                                        let mut st = shared.borrow_mut();
                                        st.in_flight = st.in_flight.saturating_sub(1);
                                        let next_attempts = attempts.saturating_add(1);
                                        let backoff = (RETRY_BASE_FRAMES << next_attempts.min(4)).min(RETRY_MAX_FRAMES);
                                        st.material_keys.insert(
                                            mat_key_owned.clone(),
                                            MaterialSlot::Failed { retry_at: now_frame + backoff, attempts: next_attempts },
                                        );
                                        st.errors.push(format!("video material {mat_key_owned}: {err:#}"));
                                        return;
                                    }
                                };

                                // Proceed to fetch the Y4M stream within the same concurrency slot.
                                {
                                    let mut st = shared.borrow_mut();
                                    if matches!(st.streams.get(&tex_key), Some(StreamSlot::Loading | StreamSlot::Ready(_))) {
                                        st.in_flight = st.in_flight.saturating_sub(1);
                                        return;
                                    }
                                    st.streams.insert(tex_key.clone(), StreamSlot::Loading);
                                }
                                fetch_stream_task(shared, assets, tex_key, looping, max_dim, now_frame, 0).await;
                            });
                            None
                        }
                    }
                }
            };

            let Some(tex_key) = resolved else { continue };
            touched_keys.insert(tex_key.clone());

            // Reserve pending video keys so the static image loader never tries to parse `.y4m` as PNG/JPEG.
            let should_retry_stream = {
                let st = self.shared.borrow();
                match st.streams.get(&tex_key) {
                    Some(StreamSlot::Ready(_)) => false,
                    Some(StreamSlot::Loading) => {
                        renderer.reserve_video_texture(&tex_key);
                        false
                    }
                    Some(StreamSlot::Failed { retry_at, attempts }) => {
                        renderer.reserve_video_texture(&tex_key);
                        now_frame >= *retry_at && st.in_flight < MAX_CONCURRENT_FETCHES && *attempts < 8
                    }
                    None => {
                        renderer.reserve_video_texture(&tex_key);
                        st.in_flight < MAX_CONCURRENT_FETCHES
                    }
                }
            };
            if should_retry_stream {
                let mut st = self.shared.borrow_mut();
                let attempts = match st.streams.get(&tex_key) {
                    Some(StreamSlot::Failed { attempts, .. }) => *attempts,
                    _ => 0,
                };
                st.in_flight += 1;
                st.streams.insert(tex_key.clone(), StreamSlot::Loading);
                let shared = self.shared.clone();
                let assets = self.assets.clone();
                let key_owned = tex_key.clone();
                let looping = spec.looping;
                spawn_local(async move {
                    fetch_stream_task(shared, assets, key_owned, looping, max_dim, now_frame, attempts).await;
                });
                continue;
            }

            if let Some(&pos) = seen_keys.get(&tex_key) {
                directives[pos].1 = spec;
            } else {
                seen_keys.insert(tex_key.clone(), directives.len());
                directives.push((tex_key, spec));
            }
        }

        // Apply deduplicated playback directives and upload changed frames zero-copy.
        {
            let mut st = self.shared.borrow_mut();
            for (tex_key, spec) in directives {
                let Some(StreamSlot::Ready(active)) = st.streams.get_mut(&tex_key) else {
                    continue;
                };
                active.last_used_frame = now_frame;
                active.player.looping = spec.looping;
                active.player.paused = !spec.playing;
                let seek_key = (spec.time.to_bits(), spec.speed.to_bits(), spec.looping);
                if spec.playing || active.last_seek != Some(seek_key) {
                    active.player.seek_time(f64::from(spec.time), f64::from(spec.speed));
                    active.last_seek = Some(seek_key);
                }
                let idx = active.player.current_index();
                if active.last_frame_index != Some(idx) {
                    let (w, h, cs, siting, y, u, v) = active.player.current_planes();
                    match renderer.upload_video_planes(&tex_key, w, h, cs, siting, y, u, v) {
                        Ok(()) => active.last_frame_index = Some(idx),
                        Err(err) => diagnostics.push(format!("video upload {tex_key}: {err:#}")),
                    }
                }
            }

            // Evict idle CPU + GPU streams once past EVICT_IDLE_FRAMES or above cumulative byte budget.
            let mut evict_keys: Vec<String> = st
                .streams
                .iter()
                .filter_map(|(k, slot)| match slot {
                    StreamSlot::Ready(active)
                        if !touched_keys.contains(k)
                            && now_frame.saturating_sub(active.last_used_frame) >= EVICT_IDLE_FRAMES =>
                    {
                        Some(k.clone())
                    }
                    _ => None,
                })
                .collect();
            if st.total_bytes > MAX_TOTAL_BYTES {
                let mut candidates: Vec<(String, u64)> = st
                    .streams
                    .iter()
                    .filter_map(|(k, slot)| match slot {
                        StreamSlot::Ready(active) if !touched_keys.contains(k) && !evict_keys.contains(k) => {
                            Some((k.clone(), active.last_used_frame))
                        }
                        _ => None,
                    })
                    .collect();
                candidates.sort_by_key(|&(_, last)| last);
                for (k, _) in candidates {
                    if st.total_bytes <= MAX_TOTAL_BYTES {
                        break;
                    }
                    evict_keys.push(k);
                }
            }
            for k in evict_keys {
                if let Some(StreamSlot::Ready(active)) = st.streams.remove(&k) {
                    st.total_bytes = st.total_bytes.saturating_sub(active.player.byte_len());
                }
                renderer.evict_video_texture(&k);
            }
        }

        diagnostics
    }
}

async fn fetch_stream_task(
    shared: Rc<RefCell<SharedState>>,
    assets: String,
    tex_key: String,
    looping: bool,
    max_dim: u32,
    now_frame: u64,
    attempts: u32,
) {
    let remaining_budget = {
        let st = shared.borrow();
        MAX_TOTAL_BYTES.saturating_sub(st.total_bytes).min(MAX_STREAM_BYTES)
    };
    let video_url = format!("{}/{}", assets.trim_end_matches('/'), tex_key);
    let result = if remaining_budget == 0 {
        Err(anyhow::anyhow!("video memory budget ({MAX_TOTAL_BYTES} bytes) exhausted"))
    } else {
        crate::fetch_bounded(&video_url, remaining_budget).await.and_then(|bytes| {
            anyhow::ensure!(bytes.starts_with(b"YUV4MPEG2"), "not a YUV4MPEG2 stream");
            let stream = IndexedY4mStream::parse_with_limits(bytes, ColorMatrix::Bt709, max_dim, remaining_budget)?;
            VideoPlayer::from_indexed_y4m(stream, looping)
        })
    };
    let mut st = shared.borrow_mut();
    st.in_flight = st.in_flight.saturating_sub(1);
    match result {
        Ok(player) => {
            st.total_bytes = st.total_bytes.saturating_add(player.byte_len());
            st.streams.insert(
                tex_key,
                StreamSlot::Ready(ActiveVideo {
                    player,
                    last_frame_index: None,
                    last_seek: None,
                    last_used_frame: now_frame,
                }),
            );
        }
        Err(err) => {
            let next_attempts = attempts.saturating_add(1);
            let backoff = (RETRY_BASE_FRAMES << next_attempts.min(4)).min(RETRY_MAX_FRAMES);
            st.streams.insert(
                tex_key.clone(),
                StreamSlot::Failed { retry_at: now_frame + backoff, attempts: next_attempts },
            );
            st.errors.push(format!("video stream {tex_key}: {err:#}"));
        }
    }
}
