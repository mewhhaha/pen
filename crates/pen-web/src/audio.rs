//! WebAudio graph, with the native mix's buses, voice fades and shared reverb.
use crate::program::error;
use anyhow::Result;
use js_sys::Uint8Array;
use pen_protocol::boundary::{AUDIO_FADE_MS, AUDIO_GLIDE_MS, AUDIO_OPEN};
use pen_protocol::{Bus, Catalog, RenderFrame, Sound, Voice};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use wasm_bindgen::{JsCast, closure::Closure};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{
    AudioBuffer, AudioBufferSourceNode, AudioContext, AudioParam, BiquadFilterNode, BiquadFilterType, GainNode,
    HtmlCanvasElement, StereoPannerNode,
};
const Q: f32 = -3.010_3;
fn glide(param: &AudioParam, value: f32, context: &AudioContext) {
    let _ = param.set_target_at_time(
        value,
        context.current_time() + 0.5 / f64::from(context.sample_rate()),
        AUDIO_GLIDE_MS as f64 / 1000.0,
    );
}
fn gain(context: &AudioContext, value: f32) -> Result<GainNode> {
    let node = context.create_gain().map_err(error)?;
    node.gain().set_value(value);
    Ok(node)
}
fn filter(context: &AudioContext, frequency: f32) -> Result<BiquadFilterNode> {
    let node = context.create_biquad_filter().map_err(error)?;
    node.set_type(BiquadFilterType::Lowpass);
    node.q().set_value(Q);
    node.frequency().set_value(frequency.min(context.sample_rate() / 2.0));
    Ok(node)
}
struct BusNodes {
    dry: BiquadFilterNode,
    send: BiquadFilterNode,
    gain: GainNode,
    send_gain: GainNode,
    reverb: GainNode,
}
struct Playing {
    source: AudioBufferSourceNode,
    filter: BiquadFilterNode,
    fade: GainNode,
    gain: GainNode,
    pan: StereoPannerNode,
    send: GainNode,
    spec: Voice,
    ended: Rc<RefCell<bool>>,
    leaving: Option<f64>,
    ramp: (f32, f32, f64, f64),
}
impl Playing {
    #[allow(clippy::cast_possible_truncation, reason = "a gain")]
    fn fade_to(&mut self, target: f32, now: f64, seconds: f64) {
        let (from, to, start, length) = self.ramp;
        let mut value = to;
        if length > 0.0 && now < start + length {
            value = from + (to - from) * ((now - start) / length).clamp(0.0, 1.0) as f32;
        }
        let param = self.fade.gain();
        let _ = param.cancel_scheduled_values(now);
        let _ = param.set_value_at_time(value, now);
        let _ = param.linear_ramp_to_value_at_time(target, now + seconds);
        self.ramp = (value, target, now, seconds);
    }
}
pub struct Audio {
    context: Option<AudioContext>,
    master: Option<BusNodes>,
    reverb_in: Option<GainNode>,
    reverb_return: Option<GainNode>,
    buses: Vec<BusNodes>,
    clips: Rc<RefCell<HashMap<u32, AudioBuffer>>>,
    loading: HashSet<u32>,
    voices: HashMap<u32, Playing>,
    assets: String,
    pending_sounds: Vec<Sound>,
    retiring: Vec<Playing>,
}
impl Audio {
    pub fn new(canvas: &HtmlCanvasElement, assets: String) -> Self {
        let mut audio = Self {
            context: None,
            master: None,
            reverb_in: None,
            reverb_return: None,
            buses: Vec::new(),
            clips: Rc::new(RefCell::new(HashMap::new())),
            loading: HashSet::new(),
            voices: HashMap::new(),
            assets,
            pending_sounds: Vec::new(),
            retiring: Vec::new(),
        };
        match audio.setup() {
            Ok(()) => {}
            Err(error) => web_sys::console::warn_1(&format!("audio unavailable: {error}").into()),
        }
        if let Some(context) = audio.context.clone() {
            let callback = Closure::<dyn FnMut(web_sys::Event)>::new(move |_| {
                let _ = context.resume();
            });
            let _ = canvas.add_event_listener_with_callback("mousedown", callback.as_ref().unchecked_ref());
            let _ = canvas.add_event_listener_with_callback("keydown", callback.as_ref().unchecked_ref());
            callback.forget();
        }
        audio
    }
    fn setup(&mut self) -> Result<()> {
        let context = AudioContext::new().map_err(error)?;
        let dry = filter(&context, AUDIO_OPEN as f32)?;
        let output = gain(&context, 1.0)?;
        dry.connect_with_audio_node(&output).map_err(error)?;
        output.connect_with_audio_node(&context.destination()).map_err(error)?;
        let input = gain(&context, 1.0)?;
        let convolver = context.create_convolver().map_err(error)?;
        convolver.set_normalize(false);
        let channels = pen_runtime::reverb::impulse_response(f64::from(context.sample_rate()));
        let buffer =
            context.create_buffer(2, u32::try_from(channels[0].len())?, context.sample_rate()).map_err(error)?;
        buffer.copy_to_channel(&channels[0], 0).map_err(error)?;
        buffer.copy_to_channel(&channels[1], 1).map_err(error)?;
        convolver.set_buffer(Some(&buffer));
        let return_gain = gain(&context, 1.0)?;
        input.connect_with_audio_node(&convolver).map_err(error)?;
        convolver.connect_with_audio_node(&return_gain).map_err(error)?;
        return_gain.connect_with_audio_node(&dry).map_err(error)?;
        let send = filter(&context, AUDIO_OPEN as f32)?;
        send.connect_with_audio_node(&input).map_err(error)?;
        self.master =
            Some(BusNodes { dry, send, gain: output.clone(), send_gain: output, reverb: return_gain.clone() });
        self.reverb_in = Some(input);
        self.reverb_return = Some(return_gain);
        self.context = Some(context);
        Ok(())
    }
    fn bus(&self, index: usize) -> &BusNodes {
        if index == 0 {
            return self.master.as_ref().unwrap();
        }
        &self.buses[index - 1]
    }
    pub fn reload(&mut self) {
        self.clips.borrow_mut().clear();
        self.loading.clear();
    }
    fn ensure_bus(&mut self, index: usize) -> Result<()> {
        let Some(context) = &self.context else {
            return Ok(());
        };
        while self.buses.len() < index {
            let dry = filter(context, AUDIO_OPEN as f32)?;
            let send = filter(context, AUDIO_OPEN as f32)?;
            let level = gain(context, 1.0)?;
            let send_gain = gain(context, 1.0)?;
            let reverb = gain(context, 0.0)?;
            dry.connect_with_audio_node(&level).map_err(error)?;
            level.connect_with_audio_node(&self.master.as_ref().unwrap().dry).map_err(error)?;
            level.connect_with_audio_node(&reverb).map_err(error)?;
            reverb.connect_with_audio_node(self.reverb_in.as_ref().unwrap()).map_err(error)?;
            send.connect_with_audio_node(&send_gain).map_err(error)?;
            send_gain.connect_with_audio_node(self.reverb_in.as_ref().unwrap()).map_err(error)?;
            self.buses.push(BusNodes { dry, send, gain: level, send_gain, reverb });
        }
        Ok(())
    }
    fn load(&mut self, id: u32, catalog: &Catalog) {
        if self.clips.borrow().contains_key(&id) || !self.loading.insert(id) {
            return;
        }
        let Some(context) = self.context.clone() else {
            return;
        };
        let clips = self.clips.clone();
        let path = format!("{}/{}", self.assets, catalog.sound(id));
        spawn_local(async move {
            let result = async {
                let bytes = crate::fetch(&path).await?;
                let promise = context.decode_audio_data(&Uint8Array::from(bytes.as_slice()).buffer()).map_err(error)?;
                let buffer: AudioBuffer = JsFuture::from(promise).await.map_err(error)?.dyn_into().map_err(error)?;
                clips.borrow_mut().insert(id, buffer);
                Ok::<_, anyhow::Error>(())
            }
            .await;
            if let Err(error) = result {
                web_sys::console::warn_1(&format!("sound {path}: {error}").into());
            }
        });
    }
    fn start_voice(&mut self, spec: Voice) -> Result<Option<Playing>> {
        self.ensure_bus(spec.bus as usize)?;
        let Some(buffer) = self.clips.borrow().get(&spec.clip).cloned() else {
            return Ok(None);
        };
        let context = self.context.as_ref().unwrap();
        let source = context.create_buffer_source().map_err(error)?;
        source.set_buffer(Some(&buffer));
        source.set_loop(spec.looping);
        source.playback_rate().set_value(spec.pitch);
        let lowpass = filter(context, spec.lowpass)?;
        let fade = gain(context, 0.0)?;
        let level = gain(context, spec.gain)?;
        let pan = context.create_stereo_panner().map_err(error)?;
        pan.pan().set_value(spec.pan.clamp(-1.0, 1.0));
        let send = gain(context, spec.reverb)?;
        source.connect_with_audio_node(&lowpass).map_err(error)?;
        lowpass.connect_with_audio_node(&fade).map_err(error)?;
        fade.connect_with_audio_node(&level).map_err(error)?;
        level.connect_with_audio_node(&pan).map_err(error)?;
        let bus = self.bus(spec.bus as usize);
        pan.connect_with_audio_node(&bus.dry).map_err(error)?;
        pan.connect_with_audio_node(&send).map_err(error)?;
        send.connect_with_audio_node(&bus.send).map_err(error)?;
        let ended = Rc::new(RefCell::new(false));
        let state = ended.clone();
        let callback = Closure::once_into_js(move || *state.borrow_mut() = true);
        source.unchecked_ref::<web_sys::AudioScheduledSourceNode>().set_onended(Some(callback.unchecked_ref()));
        let now = context.current_time();
        fade.gain().set_value_at_time(0.0, now).map_err(error)?;
        fade.gain().linear_ramp_to_value_at_time(1.0, now + f64::from(spec.fade_in)).map_err(error)?;
        let mut offset = f64::from(spec.start);
        if spec.looping && buffer.duration() > 0.0 {
            offset %= buffer.duration();
        }
        source.start_with_when_and_grain_offset(now, offset).map_err(error)?;
        Ok(Some(Playing {
            source,
            filter: lowpass,
            fade,
            gain: level,
            pan,
            send,
            spec,
            ended,
            leaving: None,
            ramp: (0.0, 1.0, now, f64::from(spec.fade_in)),
        }))
    }
    pub async fn submit(&mut self, frame: &RenderFrame, catalog: &Catalog) {
        if self.context.is_none() {
            return;
        }
        self.pending_sounds.extend_from_slice(&frame.sounds);
        for id in frame.sounds.iter().map(|s| s.clip).chain(frame.voices.iter().map(|v| v.clip)) {
            self.load(id, catalog);
        }
        let result = self.mix(frame);
        if let Err(error) = result {
            web_sys::console::warn_1(&format!("mix: {error}").into());
        }
    }
    fn mix(&mut self, frame: &RenderFrame) -> Result<()> {
        let needed = frame
            .buses
            .len()
            .max(1)
            .max(frame.voices.iter().map(|v| v.bus as usize + 1).max().unwrap_or(1))
            .max(frame.sounds.iter().map(|v| v.bus as usize + 1).max().unwrap_or(1));
        self.ensure_bus(needed - 1)?;
        let context = self.context.as_ref().unwrap().clone();
        let now = context.current_time();
        for index in 0..=self.buses.len() {
            let mut fallback = Bus { gain: 1.0, lowpass: AUDIO_OPEN as f32, reverb: 0.0 };
            if index == 0 {
                fallback.reverb = 1.0;
            }
            let spec = frame.buses.get(index).unwrap_or(&fallback);
            let bus = self.bus(index);
            glide(&bus.gain.gain(), spec.gain, &context);
            if index != 0 {
                glide(&bus.send_gain.gain(), spec.gain, &context);
            }
            glide(&bus.dry.frequency(), spec.lowpass.min(context.sample_rate() / 2.0), &context);
            if index != 0 {
                glide(&bus.send.frequency(), spec.lowpass.min(context.sample_rate() / 2.0), &context);
            }
            glide(&bus.reverb.gain(), spec.reverb, &context);
        }
        self.retiring.retain(|playing| !*playing.ended.borrow());
        self.voices.retain(|_, playing| {
            if let Some(until) = playing.leaving {
                return now < until;
            }
            true
        });
        for (&id, playing) in &mut self.voices {
            if !frame.voices.iter().any(|v| v.id == id) && playing.leaving.is_none() {
                let seconds = f64::from(playing.spec.fade_out).max(AUDIO_FADE_MS as f64 / 1000.0);
                playing.fade_to(0.0, now, seconds);

                let _ =
                    playing.source.unchecked_ref::<web_sys::AudioScheduledSourceNode>().stop_with_when(now + seconds);
                playing.leaving = Some(now + seconds);
            }
        }
        for spec in &frame.voices {
            let changed = self.voices.get(&spec.id).is_some_and(|p| p.spec.clip != spec.clip || p.spec.bus != spec.bus);
            if changed && let Some(mut p) = self.voices.remove(&spec.id) {
                let seconds = f64::from(p.spec.fade_out).max(AUDIO_FADE_MS as f64 / 1000.0);
                p.fade_to(0.0, now, seconds);
                let _ = p.source.unchecked_ref::<web_sys::AudioScheduledSourceNode>().stop_with_when(now + seconds);
                self.retiring.push(p);
            }
            if let Some(playing) = self.voices.get_mut(&spec.id) {
                if playing.leaving.take().is_some() {
                    let _ = playing
                        .source
                        .unchecked_ref::<web_sys::AudioScheduledSourceNode>()
                        .stop_with_when(now + 1_000_000.0);
                    playing.fade_to(1.0, now, f64::from(spec.fade_in).max(AUDIO_FADE_MS as f64 / 1000.0));
                }
                if !*playing.ended.borrow() {
                    glide(&playing.gain.gain(), spec.gain, &context);
                    glide(&playing.pan.pan(), spec.pan.clamp(-1.0, 1.0), &context);
                    glide(&playing.source.playback_rate(), spec.pitch, &context);
                    glide(&playing.filter.frequency(), spec.lowpass.min(context.sample_rate() / 2.0), &context);
                    glide(&playing.send.gain(), spec.reverb, &context);
                    playing.spec = *spec;
                }
            } else if let Some(playing) = self.start_voice(*spec)? {
                self.voices.insert(spec.id, playing);
            }
        }
        let mut waiting = Vec::new();
        for sound in std::mem::take(&mut self.pending_sounds) {
            let Sound { clip, bus, gain, pan, pitch, lowpass, reverb } = sound;
            match self.start_voice(Voice { clip, bus, gain, pan, pitch, lowpass, reverb, ..Voice::default() })? {
                Some(playing) => self.retiring.push(playing),
                None => waiting.push(sound),
            }
        }
        self.pending_sounds = waiting;
        Ok(())
    }
}

pub async fn qualify() -> Result<()> {
    let canvas: HtmlCanvasElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .map_err(error)?
        .dyn_into()
        .map_err(|value| error(value.into()))?;
    let mut audio = Audio::new(&canvas, String::new());
    let context = audio.context.as_ref().ok_or_else(|| anyhow::anyhow!("WebAudio context unavailable"))?;
    let buffer = context.create_buffer(1, 128, context.sample_rate()).map_err(error)?;
    let tone: Vec<f32> = (0..128).map(|index| (index as f32 * 0.1).sin() * 0.1).collect();
    buffer.copy_to_channel(&tone, 0).map_err(error)?;
    audio.clips.borrow_mut().insert(0, buffer);
    let mut frame = RenderFrame::default();
    frame.buses.push(Bus { gain: 0.1, lowpass: 5000.0, reverb: 0.1 });
    frame.voices.push(Voice {
        id: 1,
        clip: 0,
        gain: 0.1,
        pitch: 1.0,
        pan: 0.2,
        lowpass: 4000.0,
        reverb: 0.2,
        looping: true,
        fade_in: 0.01,
        fade_out: 0.01,
        ..Voice::default()
    });
    frame.sounds.push(Sound {
        clip: 0,
        gain: 0.1,
        pitch: 0.9,
        pan: -0.2,
        lowpass: 3000.0,
        reverb: 0.3,
        ..Sound::default()
    });
    audio.pending_sounds.extend_from_slice(&frame.sounds);
    audio.mix(&frame)?;
    frame.voices[0].gain = 0.2;
    frame.voices[0].pitch = 1.1;
    audio.mix(&frame)?;
    frame.voices.clear();
    audio.mix(&frame)?;
    let _ = audio.context.as_ref().unwrap().close();
    Ok(())
}
