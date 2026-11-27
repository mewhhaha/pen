//! Lean WebAudio backend compiled when `feature = "audio-advanced"` is disabled.
//! Plays catalog clips, looping voices, bus gain, pan, and pitch lazily on first
//! sound/voice without linking convolver reverb, 3-band EQ, compressors, 3D HRTF
//! panners, analysers, media-element streaming, or synth oscillators.
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use anyhow::Result;
use js_sys::Uint8Array;
use pen_protocol::{Catalog, RenderFrame, Voice};
use wasm_bindgen::{JsCast, closure::Closure};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{
    AudioBuffer, AudioBufferSourceNode, AudioContext, AudioContextOptions, AudioContextState, AudioNode, GainNode,
    HtmlCanvasElement, StereoPannerNode,
};

use crate::program::error;

struct Playing {
    source: AudioBufferSourceNode,
    gain: GainNode,
    pan: StereoPannerNode,
    clip: u32,
    bus: u32,
    ended: Rc<RefCell<bool>>,
}

pub struct Audio {
    canvas: HtmlCanvasElement,
    assets: String,
    context: Option<AudioContext>,
    master: Option<GainNode>,
    buses: Vec<GainNode>,
    clips: Rc<RefCell<HashMap<u32, AudioBuffer>>>,
    loading: HashSet<u32>,
    voices: HashMap<u32, Playing>,
    failed: bool,
}

impl Audio {
    pub fn new(canvas: &HtmlCanvasElement, assets: String) -> Self {
        Self {
            canvas: canvas.clone(),
            assets,
            context: None,
            master: None,
            buses: Vec::new(),
            clips: Rc::new(RefCell::new(HashMap::new())),
            loading: HashSet::new(),
            voices: HashMap::new(),
            failed: false,
        }
    }

    fn ensure_context(&mut self) -> Option<AudioContext> {
        if let Some(ctx) = &self.context {
            return Some(ctx.clone());
        }
        if self.failed {
            return None;
        }
        let options = AudioContextOptions::new();
        let Ok(ctx) = AudioContext::new_with_context_options(&options) else {
            self.failed = true;
            return None;
        };
        let Ok(master) = ctx.create_gain() else {
            self.failed = true;
            return None;
        };
        let _ = master.connect_with_audio_node(&ctx.destination());
        let resume_ctx = ctx.clone();
        let callback = Closure::<dyn FnMut(web_sys::Event)>::new(move |_| {
            let _ = resume_ctx.resume();
        });
        let _ = self.canvas.add_event_listener_with_callback("mousedown", callback.as_ref().unchecked_ref());
        let _ = self.canvas.add_event_listener_with_callback("keydown", callback.as_ref().unchecked_ref());
        callback.forget();
        self.master = Some(master);
        self.context = Some(ctx.clone());
        Some(ctx)
    }

    fn bus_node(&mut self, index: u32) -> Option<GainNode> {
        let ctx = self.context.as_ref()?.clone();
        let master = self.master.as_ref()?.clone();
        if index == 0 {
            return Some(master);
        }
        let slot = index as usize - 1;
        while self.buses.len() <= slot {
            let Ok(g) = ctx.create_gain() else { return None };
            let _ = g.connect_with_audio_node(&master);
            self.buses.push(g);
        }
        Some(self.buses[slot].clone())
    }

    fn load(&mut self, id: u32, catalog: &Catalog) {
        if self.clips.borrow().contains_key(&id) || !self.loading.insert(id) {
            return;
        }
        let Some(context) = self.context.clone() else { return };
        let path = format!("{}/{}", self.assets, catalog.sound(id));
        let clips = self.clips.clone();
        spawn_local(async move {
            let decoded: Result<AudioBuffer> = async {
                let bytes = crate::fetch(&path).await?;
                if path.ends_with(".sound") {
                    #[cfg(feature = "sound-recipes")]
                    {
                        let recipe = pen_runtime::sound_recipe::Recipe::parse(&bytes)?;
                        let samples = recipe.render()?;
                        let buffer = context
                            .create_buffer(1, u32::try_from(samples.len())?, recipe.rate as f32)
                            .map_err(error)?;
                        buffer.copy_to_channel(&samples, 0).map_err(error)?;
                        return Ok(buffer);
                    }
                    #[cfg(not(feature = "sound-recipes"))]
                    anyhow::bail!("use a host built with sound-recipes for .sound assets");
                }
                let array = Uint8Array::from(bytes.as_slice());
                let promise = context.decode_audio_data(&array.buffer()).map_err(error)?;
                Ok(JsFuture::from(promise).await.map_err(error)?.unchecked_into())
            }
            .await;
            if let Ok(buf) = decoded {
                clips.borrow_mut().insert(id, buf);
            }
        });
    }

    #[cfg_attr(not(feature = "devtools"), allow(dead_code))]
    pub fn reload(&mut self) {
        self.clips.borrow_mut().clear();
        self.loading.clear();
    }

    pub async fn submit(&mut self, frame: &RenderFrame, catalog: &Catalog) {
        if frame.sounds.is_empty() && frame.voices.is_empty() && self.voices.is_empty() {
            return;
        }
        let Some(ctx) = self.ensure_context() else { return };
        if ctx.state() == AudioContextState::Suspended {
            let _ = ctx.resume();
        }
        for id in frame.sounds.iter().map(|s| s.clip).chain(frame.voices.iter().map(|v| v.clip)) {
            self.load(id, catalog);
        }

        // Update bus volumes.
        for (index, bus) in frame.buses.iter().enumerate() {
            if let Ok(bus_id) = u32::try_from(index)
                && let Some(node) = self.bus_node(bus_id)
            {
                node.gain().set_value(bus.gain.max(0.0));
            }
        }

        // One-shot sounds.
        for sound in &frame.sounds {
            let Some(buf) = self.clips.borrow().get(&sound.clip).cloned() else { continue };
            let Some(dest) = self.bus_node(sound.bus) else { continue };
            let Ok(src) = ctx.create_buffer_source() else { continue };
            let Ok(gain) = ctx.create_gain() else { continue };
            let Ok(pan) = ctx.create_stereo_panner() else { continue };
            src.set_buffer(Some(&buf));
            src.playback_rate().set_value(sound.pitch.max(0.01));
            gain.gain().set_value(sound.gain.max(0.0));
            pan.pan().set_value(sound.pan.clamp(-1.0, 1.0));
            let _ = src.connect_with_audio_node(&gain);
            let _ = gain.connect_with_audio_node(&pan);
            let _ = pan.connect_with_audio_node(&dest);
            let _ = src.start();
        }

        // Persistent voices.
        let active_ids: HashSet<u32> = frame.voices.iter().map(|v| v.id).collect();
        self.voices.retain(|id, playing| {
            let keep = active_ids.contains(id) && !*playing.ended.borrow();
            if !keep {
                let _ = playing.source.unchecked_ref::<web_sys::AudioScheduledSourceNode>().stop();
                let _: &AudioNode = playing.source.as_ref();
            }
            keep
        });
        for &spec in &frame.voices {
            if let Some(playing) = self.voices.get(&spec.id)
                && playing.clip == spec.clip
                && playing.bus == spec.bus
            {
                playing.gain.gain().set_value(spec.gain.max(0.0));
                playing.pan.pan().set_value(spec.pan.clamp(-1.0, 1.0));
                playing.source.playback_rate().set_value(spec.pitch.max(0.01));
                continue;
            }
            if let Some(old) = self.voices.remove(&spec.id) {
                let _ = old.source.unchecked_ref::<web_sys::AudioScheduledSourceNode>().stop();
            }
            if let Some(playing) = self.start_voice(spec) {
                self.voices.insert(spec.id, playing);
            }
        }
    }

    fn start_voice(&mut self, spec: Voice) -> Option<Playing> {
        let ctx = self.context.as_ref()?.clone();
        let buf = self.clips.borrow().get(&spec.clip)?.clone();
        let dest = self.bus_node(spec.bus)?;
        let src = ctx.create_buffer_source().ok()?;
        let gain = ctx.create_gain().ok()?;
        let pan = ctx.create_stereo_panner().ok()?;
        src.set_buffer(Some(&buf));
        src.set_loop(spec.looping);
        src.playback_rate().set_value(spec.pitch.max(0.01));
        gain.gain().set_value(spec.gain.max(0.0));
        pan.pan().set_value(spec.pan.clamp(-1.0, 1.0));
        let _ = src.connect_with_audio_node(&gain);
        let _ = gain.connect_with_audio_node(&pan);
        let _ = pan.connect_with_audio_node(&dest);
        let ended = Rc::new(RefCell::new(false));
        let flag = ended.clone();
        let cb = Closure::once_into_js(move || {
            *flag.borrow_mut() = true;
        });
        src.unchecked_ref::<web_sys::AudioScheduledSourceNode>().set_onended(Some(cb.unchecked_ref()));
        let _ = src.start();
        Some(Playing { source: src, gain, pan, clip: spec.clip, bus: spec.bus, ended })
    }

    #[cfg(feature = "devtools")]
    pub fn report(&self) -> Option<pen_runtime::mix::AudioReport> {
        None
    }

    #[cfg(feature = "devtools")]
    pub fn monitor(&mut self, _bus: usize, _mute: bool, _on: bool) -> bool {
        false
    }
}
