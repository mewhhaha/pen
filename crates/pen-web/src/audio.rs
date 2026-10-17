//! WebAudio graph, with the native mix's buses, voice fades and shared reverb.
//! A voice the frame places in the world pans through a PannerNode
//! (equal-power) in place of its StereoPannerNode, placed in the listener's
//! frame (`pen_runtime::panner`) with the AudioListener left as it starts.
//! A clip too long to hold decoded (`pen_runtime::clip::streams`, as the
//! native host decides) streams through a media element instead: its bytes
//! as a Blob URL, the element's MediaElementAudioSourceNode in place of an
//! AudioBufferSourceNode, its pitch the element's playbackRate. It starts
//! when the element can, not at a sample, but plays through the same nodes.
//! For the dev tools, each bus's output feeds two AnalyserNodes, its meters
//! over their last 2048 samples, and the developer's mute and solo
//! (`pen_runtime::mix::Monitor`) glide bus gains to 0. The page's
//! `?audio-latency=` (interactive, balanced, playback or seconds) is the
//! AudioContext's latency hint.
use crate::program::error;
use anyhow::Result;
use js_sys::Uint8Array;
use pen_protocol::boundary::{AUDIO_FADE_MS, AUDIO_GLIDE_MS, AUDIO_OPEN};
use pen_protocol::{Bus, Catalog, RenderFrame, Sound, Voice, distance_model};
use pen_runtime::mix::{AudioReport, BusReport, Monitor};
use pen_runtime::panner::{Placement, Placements};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use wasm_bindgen::{JsCast, closure::Closure};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{
    AnalyserNode, AudioBuffer, AudioBufferSourceNode, AudioContext, AudioContextOptions, AudioContextState, AudioNode,
    AudioParam, BiquadFilterNode, BiquadFilterType, Blob, BlobPropertyBag, DistanceModelType, GainNode,
    HtmlAudioElement, HtmlCanvasElement, MediaElementAudioSourceNode, PannerNode, PanningModelType, StereoPannerNode,
    Url,
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
/// Places `node` where `placement` says: its settings at once, its position
/// and orientation gliding there, or at once as it starts.
fn place(node: &PannerNode, placement: &Placement, context: &AudioContext, start: bool) {
    node.set_distance_model(match placement.distance_model {
        distance_model::LINEAR => DistanceModelType::Linear,
        distance_model::EXPONENTIAL => DistanceModelType::Exponential,
        _ => DistanceModelType::Inverse,
    });
    node.set_ref_distance(placement.ref_distance);
    node.set_max_distance(placement.max_distance);
    node.set_rolloff_factor(placement.rolloff);
    node.set_cone_inner_angle(placement.cone_inner);
    node.set_cone_outer_angle(placement.cone_outer);
    node.set_cone_outer_gain(placement.cone_outer_gain);
    let [x, y, z] = placement.position;
    let [ox, oy, oz] = placement.orientation;
    let params = [
        (node.position_x(), x),
        (node.position_y(), y),
        (node.position_z(), z),
        (node.orientation_x(), ox),
        (node.orientation_y(), oy),
        (node.orientation_z(), oz),
    ];
    for (param, value) in params {
        if start {
            param.set_value(value);
        } else {
            glide(&param, value, context);
        }
    }
}
/// How a voice is panned: by its pan, or where the frame places it.
enum Panning {
    Stereo(StereoPannerNode),
    Placed(PannerNode),
}
impl Panning {
    fn node(&self) -> &AudioNode {
        match self {
            Panning::Stereo(node) => node,
            Panning::Placed(node) => node,
        }
    }
}
/// A clip as the browser has it: decoded whole, or a long one's bytes as a
/// Blob URL a media element streams, and how long it lasts.
#[derive(Clone)]
enum Loaded {
    Buffer(AudioBuffer),
    Streamed { url: String, seconds: f64 },
}
/// A long clip's bytes, to stream.
fn streamed(bytes: &[u8]) -> Result<Loaded> {
    let seconds = pen_runtime::clip::header(bytes).map_or(0.0, |header| header.seconds());
    let options = BlobPropertyBag::new();
    options.set_type(if bytes.starts_with(b"OggS") { "audio/ogg" } else { "audio/wav" });
    let parts = js_sys::Array::of1(&Uint8Array::from(bytes));
    let blob = Blob::new_with_u8_array_sequence_and_options(&parts, &options).map_err(error)?;
    // Never revoked: an element streaming it fetches from it as it plays.
    Ok(Loaded::Streamed { url: Url::create_object_url_with_blob(&blob).map_err(error)?, seconds })
}
/// A media element streaming a voice's clip: paused and let go of its clip
/// when the voice goes.
struct Media {
    element: HtmlAudioElement,
    node: MediaElementAudioSourceNode,
    /// When it is to stop.
    stop: Cell<Option<f64>>,
}
impl Drop for Media {
    fn drop(&mut self) {
        let _ = self.element.pause();
        let _ = self.element.remove_attribute("src");
        self.element.load();
    }
}
/// What plays a voice's clip.
enum Source {
    Buffer(AudioBufferSourceNode),
    Media(Media),
}
impl Source {
    fn node(&self) -> &AudioNode {
        match self {
            Source::Buffer(node) => node,
            Source::Media(media) => &media.node,
        }
    }
    /// Stops it at `when`.
    fn stop_at(&self, when: f64) {
        match self {
            Source::Buffer(node) => {
                let _ = node.unchecked_ref::<web_sys::AudioScheduledSourceNode>().stop_with_when(when);
            }
            Source::Media(media) => media.stop.set(Some(when)),
        }
    }
    /// Plays on: no stop.
    fn keep(&self, now: f64) {
        match self {
            Source::Buffer(node) => {
                let _ = node.unchecked_ref::<web_sys::AudioScheduledSourceNode>().stop_with_when(now + 1_000_000.0);
            }
            Source::Media(media) => media.stop.set(None),
        }
    }
    fn pitch(&self, pitch: f32, context: &AudioContext) {
        match self {
            Source::Buffer(node) => glide(&node.playback_rate(), pitch, context),
            Source::Media(media) => media.element.set_playback_rate(f64::from(pitch).clamp(0.0625, 16.0)),
        }
    }
    /// A media element is due to stop by `now`; one held back (the page
    /// had not been interacted with yet) plays once the context does.
    fn stopped(&self, now: f64, context: &AudioContext) -> bool {
        let Source::Media(media) = self else { return false };
        if media.stop.get().is_some_and(|stop| now >= stop) {
            return true;
        }
        if media.element.paused() && !media.element.ended() && context.state() == AudioContextState::Running {
            let _ = media.element.play();
        }
        false
    }
}
/// A bus's output, left and right, as AnalyserNodes see it.
struct Meter {
    left: AnalyserNode,
    right: AnalyserNode,
}
impl Meter {
    fn new(context: &AudioContext, output: &AudioNode) -> Result<Meter> {
        let splitter = context.create_channel_splitter_with_number_of_outputs(2).map_err(error)?;
        let left = context.create_analyser().map_err(error)?;
        let right = context.create_analyser().map_err(error)?;
        output.connect_with_audio_node(&splitter).map_err(error)?;
        splitter.connect_with_audio_node_and_output(&left, 0).map_err(error)?;
        splitter.connect_with_audio_node_and_output(&right, 1).map_err(error)?;
        Ok(Meter { left, right })
    }
    /// The peak and RMS of the samples the analysers hold.
    #[allow(clippy::cast_possible_truncation, reason = "a level to show")]
    fn read(&self) -> ([f32; 2], [f32; 2]) {
        let mut samples = vec![0.0f32; self.left.fft_size() as usize];
        let mut side = |analyser: &AnalyserNode| {
            analyser.get_float_time_domain_data(&mut samples);
            let peak = samples.iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
            let squares: f64 = samples.iter().map(|&sample| f64::from(sample) * f64::from(sample)).sum();
            (peak, (squares / samples.len().max(1) as f64).sqrt() as f32)
        };
        let ((peak_left, rms_left), (peak_right, rms_right)) = (side(&self.left), side(&self.right));
        ([peak_left, peak_right], [rms_left, rms_right])
    }
}
struct BusNodes {
    dry: BiquadFilterNode,
    send: BiquadFilterNode,
    gain: GainNode,
    send_gain: GainNode,
    reverb: GainNode,
    meter: Meter,
}
struct Playing {
    source: Source,
    filter: BiquadFilterNode,
    fade: GainNode,
    gain: GainNode,
    pan: Panning,
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
    clips: Rc<RefCell<HashMap<u32, Loaded>>>,
    loading: HashSet<u32>,
    voices: HashMap<u32, Playing>,
    assets: String,
    pending_sounds: Vec<(Sound, Option<Placement>)>,
    retiring: Vec<Playing>,
    /// The developer's mute and solo.
    monitor: Monitor,
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
            monitor: Monitor::default(),
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
        let options = AudioContextOptions::new();
        if let Some(hint) = latency_hint() {
            options.set_latency_hint(&hint);
        }
        let context = AudioContext::new_with_context_options(&options).map_err(error)?;
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
        let meter = Meter::new(&context, &output)?;
        self.master =
            Some(BusNodes { dry, send, gain: output.clone(), send_gain: output, reverb: return_gain.clone(), meter });
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
            let meter = Meter::new(context, &level)?;
            self.buses.push(BusNodes { dry, send, gain: level, send_gain, reverb, meter });
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
                if pen_runtime::clip::streams(&bytes) {
                    clips.borrow_mut().insert(id, streamed(&bytes)?);
                    return Ok(());
                }
                let promise = context.decode_audio_data(&Uint8Array::from(bytes.as_slice()).buffer()).map_err(error)?;
                let buffer: AudioBuffer = JsFuture::from(promise).await.map_err(error)?.dyn_into().map_err(error)?;
                clips.borrow_mut().insert(id, Loaded::Buffer(buffer));
                Ok::<_, anyhow::Error>(())
            }
            .await;
            if let Err(error) = result {
                web_sys::console::warn_1(&format!("sound {path}: {error}").into());
            }
        });
    }
    fn start_voice(&mut self, spec: Voice, placement: Option<Placement>) -> Result<Option<Playing>> {
        self.ensure_bus(spec.bus as usize)?;
        let Some(loaded) = self.clips.borrow().get(&spec.clip).cloned() else {
            return Ok(None);
        };
        let context = self.context.as_ref().unwrap();
        let (source, seconds) = match &loaded {
            Loaded::Buffer(buffer) => {
                let source = context.create_buffer_source().map_err(error)?;
                source.set_buffer(Some(buffer));
                source.set_loop(spec.looping);
                source.playback_rate().set_value(spec.pitch);
                (Source::Buffer(source), buffer.duration())
            }
            Loaded::Streamed { url, seconds } => {
                let element = HtmlAudioElement::new_with_src(url).map_err(error)?;
                element.set_loop(spec.looping);
                element.set_playback_rate(f64::from(spec.pitch).clamp(0.0625, 16.0));
                // Pitch, as an AudioBufferSourceNode's playbackRate has it.
                let _ = js_sys::Reflect::set(&element, &"preservesPitch".into(), &false.into());
                let node = context.create_media_element_source(&element).map_err(error)?;
                (Source::Media(Media { element, node, stop: Cell::new(None) }), *seconds)
            }
        };
        let lowpass = filter(context, spec.lowpass)?;
        let fade = gain(context, 0.0)?;
        let level = gain(context, spec.gain)?;
        let pan = match placement {
            Some(placement) => {
                let node = context.create_panner().map_err(error)?;
                node.set_panning_model(PanningModelType::Equalpower);
                place(&node, &placement, context, true);
                Panning::Placed(node)
            }
            None => {
                let node = context.create_stereo_panner().map_err(error)?;
                node.pan().set_value(spec.pan.clamp(-1.0, 1.0));
                Panning::Stereo(node)
            }
        };
        let send = gain(context, spec.reverb)?;
        source.node().connect_with_audio_node(&lowpass).map_err(error)?;
        lowpass.connect_with_audio_node(&fade).map_err(error)?;
        fade.connect_with_audio_node(&level).map_err(error)?;
        level.connect_with_audio_node(pan.node()).map_err(error)?;
        let bus = self.bus(spec.bus as usize);
        pan.node().connect_with_audio_node(&bus.dry).map_err(error)?;
        pan.node().connect_with_audio_node(&send).map_err(error)?;
        send.connect_with_audio_node(&bus.send).map_err(error)?;
        let ended = Rc::new(RefCell::new(false));
        let state = ended.clone();
        let callback = Closure::once_into_js(move || *state.borrow_mut() = true);
        let now = context.current_time();
        fade.gain().set_value_at_time(0.0, now).map_err(error)?;
        fade.gain().linear_ramp_to_value_at_time(1.0, now + f64::from(spec.fade_in)).map_err(error)?;
        let mut offset = f64::from(spec.start);
        if spec.looping && seconds > 0.0 {
            offset %= seconds;
        }
        match &source {
            Source::Buffer(node) => {
                node.unchecked_ref::<web_sys::AudioScheduledSourceNode>().set_onended(Some(callback.unchecked_ref()));
                node.start_with_when_and_grain_offset(now, offset).map_err(error)?;
            }
            Source::Media(media) => {
                media.element.set_onended(Some(callback.unchecked_ref()));
                media.element.set_current_time(offset);
                let _ = media.element.play();
            }
        }
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
        let placements = Placements::of(frame);
        self.pending_sounds
            .extend(frame.sounds.iter().enumerate().map(|(index, sound)| (*sound, placements.sound(index))));
        for id in frame.sounds.iter().map(|s| s.clip).chain(frame.voices.iter().map(|v| v.clip)) {
            self.load(id, catalog);
        }
        let result = self.mix(frame, &placements);
        if let Err(error) = result {
            web_sys::console::warn_1(&format!("mix: {error}").into());
        }
    }
    fn mix(&mut self, frame: &RenderFrame, placements: &Placements) -> Result<()> {
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
            let level = if self.monitor.audible(index) { spec.gain } else { 0.0 };
            glide(&bus.gain.gain(), level, &context);
            if index != 0 {
                glide(&bus.send_gain.gain(), level, &context);
            }
            glide(&bus.dry.frequency(), spec.lowpass.min(context.sample_rate() / 2.0), &context);
            if index != 0 {
                glide(&bus.send.frequency(), spec.lowpass.min(context.sample_rate() / 2.0), &context);
            }
            glide(&bus.reverb.gain(), spec.reverb, &context);
        }
        self.retiring.retain(|playing| !*playing.ended.borrow() && !playing.source.stopped(now, &context));
        self.voices.retain(|_, playing| {
            if let Some(until) = playing.leaving {
                return now < until;
            }
            !playing.source.stopped(now, &context)
        });
        for (&id, playing) in &mut self.voices {
            if !frame.voices.iter().any(|v| v.id == id) && playing.leaving.is_none() {
                let seconds = f64::from(playing.spec.fade_out).max(AUDIO_FADE_MS as f64 / 1000.0);
                playing.fade_to(0.0, now, seconds);
                playing.source.stop_at(now + seconds);
                playing.leaving = Some(now + seconds);
            }
        }
        for spec in &frame.voices {
            let placement = placements.voice(spec.id);
            let changed = self.voices.get(&spec.id).is_some_and(|p| {
                p.spec.clip != spec.clip
                    || p.spec.bus != spec.bus
                    || matches!(p.pan, Panning::Placed(_)) != placement.is_some()
            });
            if changed && let Some(mut p) = self.voices.remove(&spec.id) {
                let seconds = f64::from(p.spec.fade_out).max(AUDIO_FADE_MS as f64 / 1000.0);
                p.fade_to(0.0, now, seconds);
                p.source.stop_at(now + seconds);
                self.retiring.push(p);
            }
            if let Some(playing) = self.voices.get_mut(&spec.id) {
                if playing.leaving.take().is_some() {
                    playing.source.keep(now);
                    playing.fade_to(1.0, now, f64::from(spec.fade_in).max(AUDIO_FADE_MS as f64 / 1000.0));
                }
                if !*playing.ended.borrow() {
                    glide(&playing.gain.gain(), spec.gain, &context);
                    match (&playing.pan, &placement) {
                        (Panning::Placed(node), Some(placement)) => place(node, placement, &context, false),
                        (Panning::Stereo(node), _) => glide(&node.pan(), spec.pan.clamp(-1.0, 1.0), &context),
                        (Panning::Placed(_), None) => {}
                    }
                    playing.source.pitch(spec.pitch, &context);
                    glide(&playing.filter.frequency(), spec.lowpass.min(context.sample_rate() / 2.0), &context);
                    glide(&playing.send.gain(), spec.reverb, &context);
                    playing.spec = *spec;
                }
            } else if let Some(playing) = self.start_voice(*spec, placement)? {
                self.voices.insert(spec.id, playing);
            }
        }
        let mut waiting = Vec::new();
        for (sound, placement) in std::mem::take(&mut self.pending_sounds) {
            let Sound { clip, bus, gain, pan, pitch, lowpass, reverb } = sound;
            let voice = Voice { clip, bus, gain, pan, pitch, lowpass, reverb, ..Voice::default() };
            match self.start_voice(voice, placement)? {
                Some(playing) => self.retiring.push(playing),
                None => waiting.push((sound, placement)),
            }
        }
        self.pending_sounds = waiting;
        Ok(())
    }
}

/// The page's `?audio-latency=`: a latency category, or seconds.
fn latency_hint() -> Option<wasm_bindgen::JsValue> {
    let search = web_sys::window()?.location().search().ok()?;
    let hint = web_sys::UrlSearchParams::new_with_str(&search).ok()?.get("audio-latency")?;
    match hint.parse::<f64>() {
        Ok(seconds) if seconds.is_finite() && seconds >= 0.0 => Some(seconds.into()),
        _ if ["interactive", "balanced", "playback"].contains(&hint.as_str()) => Some(hint.into()),
        _ => None,
    }
}
impl Audio {
    /// The mix as it plays, for the dev tools: how late the output is (the
    /// context's base and output latency), and each bus's meters.
    pub fn report(&self) -> Option<AudioReport> {
        let context = self.context.as_ref()?;
        let seconds = |name: &str| js_sys::Reflect::get(context, &name.into()).ok().and_then(|value| value.as_f64());
        let latency = match (seconds("baseLatency"), seconds("outputLatency")) {
            (None, None) => None,
            (base, output) => Some((base.unwrap_or(0.0) + output.unwrap_or(0.0)) * 1000.0),
        };
        let buses = (0..pen_protocol::boundary::MAX_BUSES)
            .map(|index| {
                let bit = 1u32 << index;
                let (muted, soloed) = (self.monitor.muted & bit != 0, self.monitor.soloed & bit != 0);
                match (index <= self.buses.len()).then(|| self.bus(index)) {
                    Some(bus) => {
                        let (peak, rms) = bus.meter.read();
                        BusReport { peak, rms, used: true, muted, soloed }
                    }
                    None => BusReport { muted, soloed, ..BusReport::default() },
                }
            })
            .collect();
        Some(AudioReport {
            latency_ms: latency,
            rate: f64::from(context.sample_rate()),
            buffer: None,
            voices: self.voices.len() + self.retiring.len(),
            buses,
            ..AudioReport::default()
        })
    }
    /// The developer mutes (`mute`) or solos bus `bus`, or stops, from the
    /// next frame on; false without sound.
    pub fn monitor(&mut self, bus: usize, mute: bool, on: bool) -> bool {
        self.monitor.set(bus, mute, on);
        self.context.is_some()
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
    audio.clips.borrow_mut().insert(0, Loaded::Buffer(buffer));
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
    // A second voice and the sound placed in the world, heard from a
    // listener of the frame's own.
    frame.voices.push(Voice { id: 2, ..frame.voices[0] });
    let at = |x: f32, y: f32, z: f32| pen_protocol::Vector { x, y, z };
    frame.listener.push(pen_protocol::Listener {
        position: at(1.0, 0.0, 0.0),
        forward: at(0.0, 0.0, -1.0),
        up: at(0.0, 1.0, 0.0),
    });
    let panner = pen_protocol::Panner {
        voice: 2,
        position: at(3.0, 0.0, -2.0),
        orientation: at(-1.0, 0.0, 0.0),
        distance_model: distance_model::LINEAR,
        ref_distance: 1.0,
        max_distance: 20.0,
        rolloff: 0.8,
        cone_inner: 90.0,
        cone_outer: 270.0,
        cone_outer_gain: 0.3,
        ..pen_protocol::Panner::default()
    };
    frame.panners = vec![panner, pen_protocol::Panner { voice: 0, sound: 0, ..panner }];
    let placements = Placements::of(&frame);
    audio.pending_sounds.push((frame.sounds[0], placements.sound(0)));
    audio.mix(&frame, &placements)?;
    anyhow::ensure!(
        matches!(audio.voices.get(&2).map(|playing| &playing.pan), Some(Panning::Placed(_))),
        "a placed voice must pan through a PannerNode"
    );
    frame.voices[0].gain = 0.2;
    frame.voices[0].pitch = 1.1;
    frame.panners[0].position = at(-2.0, 1.0, 1.0);
    audio.mix(&frame, &Placements::of(&frame))?;
    // No longer placed, it starts afresh with a StereoPannerNode.
    frame.panners.clear();
    audio.mix(&frame, &Placements::of(&frame))?;
    anyhow::ensure!(
        matches!(audio.voices.get(&2).map(|playing| &playing.pan), Some(Panning::Stereo(_))),
        "a voice no longer placed must start again unplaced"
    );
    // A long clip streams through a media element.
    let rate = 8000u32;
    let mut long = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
    for value in [16, 1 | 1 << 16, rate, rate * 2, 2 | 16 << 16] {
        long.extend_from_slice(&value.to_le_bytes());
    }
    let samples = rate * 13;
    long.extend_from_slice(b"data");
    long.extend_from_slice(&(samples * 2).to_le_bytes());
    long.extend((0..samples).flat_map(|index| ((index % 64) as i16 * 200).to_le_bytes()));
    anyhow::ensure!(pen_runtime::clip::streams(&long), "a 13 s clip must stream");
    audio.clips.borrow_mut().insert(1, streamed(&long)?);
    frame.voices.push(Voice { id: 3, clip: 1, start: 20.0, ..frame.voices[0] });
    audio.mix(&frame, &Placements::of(&frame))?;
    anyhow::ensure!(
        matches!(audio.voices.get(&3).map(|playing| &playing.source), Some(Source::Media(_))),
        "a long clip must stream through a media element"
    );
    frame.voices.pop();
    // The dev tools' meters and mute, on a second bus.
    frame.buses.push(Bus { gain: 0.5, lowpass: AUDIO_OPEN as f32, reverb: 0.0 });
    anyhow::ensure!(audio.monitor(1, true, true), "a host with sound mutes");
    audio.mix(&frame, &Placements::of(&frame))?;
    let report = audio.report().ok_or_else(|| anyhow::anyhow!("no audio report"))?;
    anyhow::ensure!(
        report.buses.len() == pen_protocol::boundary::MAX_BUSES && report.buses[1].used && report.buses[1].muted,
        "the audio report must have every bus, bus 1 in use and muted"
    );
    frame.voices.clear();
    audio.mix(&frame, &Placements::of(&frame))?;
    let _ = audio.context.as_ref().unwrap().close();
    Ok(())
}
