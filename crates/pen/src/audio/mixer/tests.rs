use pen_runtime::panner::{Ear, Vec3};

use super::*;

const RATE: f64 = 48_000.0;

/// A mono sine at `frequency`, `seconds` long, at `RATE`.
fn sine(frequency: f64, seconds: f64) -> Arc<Clip> {
    let frames: Arc<[[f32; 2]]> = (0..(seconds * RATE) as usize)
        .map(|i| {
            let value = (std::f64::consts::TAU * frequency * i as f64 / RATE).sin() as f32;
            [value, value]
        })
        .collect();
    Arc::new(Clip::new(frames, RATE))
}

fn voice(id: u32, clip: &Arc<Clip>) -> VoiceSpec {
    VoiceSpec {
        id,
        clip: clip.clone(),
        bus: 0,
        gain: 1.0,
        pan: 0.0,
        pitch: 1.0,
        lowpass: OPEN,
        reverb: 0.0,
        looping: true,
        start: 0.0,
        fade_in: 0.0,
        fade_out: 0.0,
        panner: None,
    }
}

/// Frames of 800 samples (60 per second): each frame's mix, then its sound.
fn play(mixer: &mut Mixer, frames: impl IntoIterator<Item = Mix>) -> Vec<[f32; 2]> {
    let mut out = Vec::new();
    for mix in frames {
        mixer.apply(&mix);
        let mut chunk = vec![[0.0f32; 2]; 800];
        mixer.render(&mut chunk);
        out.extend(chunk);
    }
    out
}

fn rms(samples: &[[f32; 2]]) -> f64 {
    (samples.iter().map(|[l, r]| f64::from(*l).powi(2) + f64::from(*r).powi(2)).sum::<f64>()
        / (2 * samples.len()) as f64)
        .sqrt()
}

fn largest_step(samples: &[[f32; 2]]) -> f32 {
    samples
        .windows(2)
        .map(|pair| (pair[1][0] - pair[0][0]).abs().max((pair[1][1] - pair[0][1]).abs()))
        .fold(0.0, f32::max)
}

#[test]
fn glides_settle_where_webaudio_does() {
    let mut glide = Glide::at(0.0);
    glide.set(1.0);
    let step = 1.0 - (-1.0 / (GLIDE_SECONDS * RATE)).exp();
    let mut value = 0.0;
    // One time constant on: 1 - 1/e of the way.
    for _ in 0..=(GLIDE_SECONDS * RATE) as usize {
        value = glide.next(step);
    }
    assert!((f64::from(value) - (1.0 - (-1.0f64).exp())).abs() < 1e-3, "{value}");
    for _ in 0..20_000 {
        glide.next(step);
    }
    assert_eq!(glide.value, 1.0);
}

#[test]
fn mono_and_stereo_panning_follow_webaudio() {
    // Mono in the middle is -3 dB a side; stereo in the middle passes as it is.
    let (l, r) = mono_gains(0.0);
    assert!((l - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6 && (r - l).abs() < 1e-6);
    assert_eq!(pan_gains(0.0), (std::f32::consts::FRAC_PI_2.cos(), 1.0));
    let (l, r) = mono_gains(1.0);
    assert!(l.abs() < 1e-6 && (r - 1.0).abs() < 1e-6);
}

#[test]
fn lowpass_passes_low_and_stops_high() {
    let mut filter = Lowpass::new(1000.0, RATE);
    let tone = |filter: &mut Lowpass, frequency: f64| {
        let mut peak = 0.0f32;
        for i in 0..48_000 {
            let y = filter.run(0, (std::f64::consts::TAU * frequency * i as f64 / RATE).sin() as f32);
            if i > 24_000 {
                peak = peak.max(y.abs());
            }
        }
        peak
    };
    assert!((tone(&mut filter, 100.0) - 1.0).abs() < 0.01);
    // A Butterworth is 3 dB down at its cutoff, 12 dB an octave past it.
    assert!((tone(&mut filter, 1000.0) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01);
    assert!(tone(&mut filter, 8000.0) < 0.02);
    let open = Lowpass::new(OPEN, RATE);
    assert_eq!((open.b, open.a), ([1.0, 0.0, 0.0], [0.0, 0.0]));
}

/// New ids start, missing ones fade out over their fade_out and go, and
/// one that comes back while fading returns where it is.
#[test]
fn voices_reconcile_by_id() {
    let clip = sine(220.0, 1.0);
    let mut mixer = Mixer::new(RATE);
    let mut faded = voice(7, &clip);
    faded.fade_out = 0.5;
    let with = || Mix { voices: vec![faded.clone()], ..Mix::default() };
    let out = play(&mut mixer, [with(), with()]);
    assert_eq!(mixer.playing(), 1);
    // Mono at the middle: 1/√2 a side, a sine's RMS 1/√2 of that.
    assert!((rms(&out[400..]) - 0.5).abs() < 0.01, "{}", rms(&out[400..]));
    // Left out: it fades over half a second, linearly.
    let out = play(&mut mixer, (0..15).map(|_| Mix::default()));
    assert_eq!(mixer.playing(), 1);
    let early = rms(&out[..2400]);
    let late = rms(&out[9600..12000]);
    assert!((early - 0.5 * (1.0 - 1200.0 / 24000.0)).abs() < 0.01, "{early}");
    assert!((late - 0.5 * (1.0 - 10800.0 / 24000.0)).abs() < 0.01, "{late}");
    // Back again: it returns from where its fade had got to, not from
    // its start, so the sine carries on without a jump.
    let before = mixer.voices[0].position;
    let out = play(&mut mixer, [with()]);
    assert!(mixer.voices[0].position > before);
    assert!(largest_step(&out) < 0.05, "{}", largest_step(&out));
    // Left out for good: gone once the fade is over.
    play(&mut mixer, (0..40).map(|_| Mix::default()));
    assert_eq!(mixer.playing(), 0);
}

#[test]
fn changes_glide_without_clicks() {
    let clip = sine(330.0, 1.0);
    let mut mixer = Mixer::new(RATE);
    // Gain, pan and lowpass jumping every frame.
    let frames = (0..120).map(|frame| {
        let mut spec = voice(1, &clip);
        let up = frame % 2 == 0;
        spec.gain = if up { 0.9 } else { 0.1 };
        spec.pan = if up { -0.8 } else { 0.8 };
        spec.lowpass = if up { 400.0 } else { OPEN };
        Mix { voices: vec![spec], ..Mix::default() }
    });
    let out = play(&mut mixer, frames);
    // A 330 Hz sine at full scale moves at most 2π·330/48000 ≈ 0.043 a sample.
    assert!(largest_step(&out) < 0.05, "{}", largest_step(&out));
}

/// A new clip for an id crossfades: the old fades out as the new comes in.
#[test]
fn clip_changes_crossfade() {
    let (low, high) = (sine(200.0, 2.0), sine(3000.0, 2.0));
    let mut mixer = Mixer::new(RATE);
    let track = |clip: &Arc<Clip>| {
        let mut spec = voice(1, clip);
        spec.fade_in = 0.5;
        spec.fade_out = 0.5;
        Mix { voices: vec![spec], ..Mix::default() }
    };
    play(&mut mixer, (0..60).map(|_| track(&low)));
    let out = play(&mut mixer, (0..60).map(|_| track(&high)));
    assert_eq!(mixer.playing(), 1);
    // Halfway through, both are there at about half.
    let middle = &out[12_000 - 960..12_000 + 960];
    let level = rms(middle);
    assert!((0.25..0.45).contains(&level), "{level}");
    assert!(largest_step(&out) < 0.45);
}

#[test]
fn a_voice_that_does_not_loop_plays_once() {
    let clip = sine(440.0, 0.1);
    let mut mixer = Mixer::new(RATE);
    let mut once = voice(3, &clip);
    once.looping = false;
    let with = || Mix { voices: vec![once.clone()], ..Mix::default() };
    let out = play(&mut mixer, (0..30).map(|_| with()));
    assert!(rms(&out[..4800]) > 0.4);
    assert_eq!(rms(&out[6000..]), 0.0);
    assert_eq!(mixer.playing(), 1);
    // Left out, then declared again: it plays again.
    play(&mut mixer, [Mix::default()]);
    assert_eq!(mixer.playing(), 0);
    let out = play(&mut mixer, [with()]);
    assert!(rms(&out) > 0.4);
}

#[test]
fn duplicate_ids_play_the_first() {
    let (a, b) = (sine(200.0, 1.0), sine(900.0, 1.0));
    let mut mixer = Mixer::new(RATE);
    play(&mut mixer, [Mix { voices: vec![voice(5, &a), voice(5, &b)], ..Mix::default() }]);
    assert_eq!(mixer.playing(), 1);
    assert!(Arc::ptr_eq(&mixer.voices[0].clip, &a));
}

/// Buses scale and filter their voices, and send to the reverb, which
/// comes back through the master.
#[test]
fn buses_and_the_reverb() {
    let clip = sine(500.0, 1.0);
    let on_bus = |bus: usize, reverb: f32, buses: Vec<BusSpec>| {
        let mut spec = voice(1, &clip);
        spec.bus = bus;
        spec.reverb = reverb;
        Mix { voices: vec![spec], buses, ..Mix::default() }
    };
    let master = BusSpec { gain: 1.0, lowpass: OPEN, reverb: 1.0, ..PLAIN };
    let mut dry = Mixer::new(RATE);
    let plain = rms(&play(&mut dry, (0..30).map(|_| on_bus(1, 0.0, vec![master, PLAIN])))[4800..]);
    let mut quiet = Mixer::new(RATE);
    let half =
        rms(&play(&mut quiet, (0..30).map(|_| on_bus(1, 0.0, vec![master, BusSpec { gain: 0.5, ..PLAIN }])))[4800..]);
    assert!((half / plain - 0.5).abs() < 0.01, "{half} {plain}");
    let mut dull = Mixer::new(RATE);
    let filtered =
        rms(&play(&mut dull, (0..30).map(|_| on_bus(1, 0.0, vec![master, BusSpec { lowpass: 250.0, ..PLAIN }])))
            [4800..]);
    assert!(filtered < plain * 0.35, "{filtered} {plain}");
    // A send adds the reverb; a master returning none takes it away.
    let mut wet = Mixer::new(RATE);
    let reverberant = rms(&play(&mut wet, (0..30).map(|_| on_bus(1, 1.0, vec![master, PLAIN])))[4800..]);
    assert!(reverberant > plain * 1.1, "{reverberant} {plain}");
    let mut returned = Mixer::new(RATE);
    let none =
        rms(&play(&mut returned, (0..30).map(|_| on_bus(1, 1.0, vec![BusSpec { reverb: 0.0, ..master }, PLAIN])))
            [4800..]);
    assert!((none - plain).abs() < 1e-4, "{none} {plain}");
}

/// The same frames give the same samples, bit for bit.
#[test]
fn mixing_is_deterministic() {
    let (a, b) = (sine(200.0, 0.7), sine(1300.0, 0.3));
    let frames = || {
        (0..90).map(|frame| {
            let mut spec = voice(1 + frame / 30, &a);
            spec.reverb = 0.5;
            spec.pitch = 1.0 + frame as f32 * 0.01;
            let sounds = if frame % 7 == 0 {
                vec![SoundSpec {
                    clip: b.clone(),
                    bus: 2,
                    gain: 0.5,
                    pan: 0.3,
                    pitch: 1.2,
                    lowpass: 3000.0,
                    reverb: 0.3,
                    panner: None,
                }]
            } else {
                Vec::new()
            };
            Mix {
                voices: vec![spec],
                sounds,
                buses: vec![MASTER, PLAIN, BusSpec { gain: 0.7, lowpass: 5000.0, reverb: 0.2, ..PLAIN }],
                ..Mix::default()
            }
        })
    };
    let first = play(&mut Mixer::new(RATE), frames());
    let second = play(&mut Mixer::new(RATE), frames());
    assert!(
        first.iter().zip(&second).all(|(a, b)| a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits())
    );
}
/// Each bus's meter is its output's peak and RMS since the last
/// reading; muting a bus silences it (gliding there) and soloing one
/// silences the others but the master.
#[test]
fn buses_are_metered_muted_and_soloed() {
    let (low, high) = (sine(200.0, 1.0), sine(900.0, 1.0));
    let frame = || {
        let mut first = voice(1, &low);
        first.bus = 1;
        let mut second = voice(2, &high);
        second.bus = 2;
        let mut master = voice(3, &low);
        master.gain = 0.5;
        Mix {
            voices: vec![first, second, master],
            buses: vec![BusSpec { reverb: 0.0, ..MASTER }, PLAIN, PLAIN],
            ..Mix::default()
        }
    };
    let mut mixer = Mixer::new(RATE);
    play(&mut mixer, (0..10).map(|_| frame()));
    let (meters, used) = mixer.take_meters();
    assert_eq!(used[..4], [true, true, true, false]);
    // A mono sine in the middle: 1/√2 a side at its peak, half that RMS.
    for bus in [1, 2] {
        assert!((meters[bus].peak[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3, "{:?}", meters[bus]);
        assert!((meters[bus].rms()[1] - 0.5).abs() < 0.01, "{:?}", meters[bus]);
    }
    assert!(meters[0].peak[0] > 1.0 && meters[3].samples == 0);
    let level = |mixer: &mut Mixer, bus: usize| {
        mixer.take_meters();
        play(mixer, (0..10).map(|_| frame()));
        mixer.take_meters().0[bus].rms()[0]
    };
    mixer.set_monitor(Monitor { muted: 0b10, soloed: 0 });
    play(&mut mixer, (0..10).map(|_| frame()));
    assert!(level(&mut mixer, 1) < 1e-4);
    assert!((level(&mut mixer, 2) - 0.5).abs() < 0.01);
    // Soloing 1 brings back none but 1 and the master's own voice, and
    // 1 is still muted.
    mixer.set_monitor(Monitor { muted: 0b10, soloed: 0b10 });
    play(&mut mixer, (0..10).map(|_| frame()));
    assert!(level(&mut mixer, 2) < 1e-4);
    let master = level(&mut mixer, 0);
    assert!((master - 0.25).abs() < 0.01, "{master}");
    mixer.set_monitor(Monitor { muted: 0, soloed: 0b10 });
    play(&mut mixer, (0..10).map(|_| frame()));
    assert!((level(&mut mixer, 1) - 0.5).abs() < 0.01);
    // Unmuted and unsoloed, everything is back, without a click.
    mixer.set_monitor(Monitor::default());
    let out = play(&mut mixer, (0..10).map(|_| frame()));
    assert!(largest_step(&out) < 0.2, "{}", largest_step(&out));
    assert!((level(&mut mixer, 2) - 0.5).abs() < 0.01);
}

/// A placed voice from where a frame puts it, with WebAudio's defaults
/// otherwise: an inverse model, ref 1, rolloff 1, no cone.
fn placed(position: Vec3) -> Placement {
    Placement::new(&pen_protocol::Panner {
        position: pen_protocol::Vector { x: position[0], y: position[1], z: position[2] },
        distance_model: pen_protocol::distance_model::INVERSE,
        ref_distance: 1.0,
        max_distance: 10_000.0,
        rolloff: 1.0,
        cone_inner: 360.0,
        cone_outer: 360.0,
        ..pen_protocol::Panner::default()
    })
}

/// A placed voice pans by where it is around the listener and quiets
/// with distance, in place of its own pan; one that loses its place
/// starts afresh, unplaced.
#[test]
fn placed_voices_pan_by_azimuth() {
    let clip = sine(300.0, 1.0);
    let level = |position: Vec3, pan: f32| {
        let mut spec = voice(1, &clip);
        spec.pan = pan;
        spec.panner = Some(placed(position));
        let out = play(&mut Mixer::new(RATE), (0..10).map(|_| Mix { voices: vec![spec.clone()], ..Mix::default() }));
        let side = |index: usize| {
            (out[2400..].iter().map(|frame| f64::from(frame[index]).powi(2)).sum::<f64>() / (out.len() - 2400) as f64)
                .sqrt()
        };
        (side(0), side(1))
    };
    // Ahead, at the reference distance: -3 dB a side, as a mono pan in
    // the middle, whatever its own pan says.
    let (left, right) = level([0.0, 0.0, -1.0], 0.9);
    let middle = 0.5;
    assert!((left - middle).abs() < 0.01 && (right - middle).abs() < 0.01, "{left} {right}");
    // To the right, all on the right; to the left behind, all on the
    // left; four units off, a quarter as loud.
    let (left, right) = level([1.0, 0.0, 0.0], 0.0);
    assert!(left < 1e-3 && (right - middle * 2f64.sqrt()).abs() < 0.01, "{left} {right}");
    let (left, right) = level([-4.0, 0.0, 0.0], 0.0);
    assert!(right < 1e-3 && (left - middle * 2f64.sqrt() / 4.0).abs() < 0.01, "{left} {right}");
    // Losing its place, it starts afresh, the old fading out.
    let mut mixer = Mixer::new(RATE);
    let mut spec = voice(1, &clip);
    spec.panner = Some(placed([1.0, 0.0, 0.0]));
    play(&mut mixer, [Mix { voices: vec![spec.clone()], ..Mix::default() }]);
    spec.panner = None;
    play(&mut mixer, [Mix { voices: vec![spec], ..Mix::default() }]);
    assert_eq!(mixer.playing(), 2);
    assert!(mixer.voices.iter().any(|voice| voice.spatial.is_none() && voice.id == Some(1)));
}

/// A source jumping across, and a listener turning, every frame glide
/// as gains do: no clicks.
#[test]
fn placed_voices_move_without_clicks() {
    let clip = sine(330.0, 1.0);
    let frames = (0..120).map(|frame| {
        let side = if frame % 2 == 0 { 3.0 } else { -3.0 };
        let turn = if frame % 3 == 0 { 0.0 } else { 1.0f32 };
        let ear = Ear {
            position: [0.0; 3],
            forward: [turn.sin(), 0.0, -turn.cos()],
            up: [0.0, 1.0, 0.0],
            velocity: [0.0; 3],
        };
        let mut spec = voice(1, &clip);
        spec.panner = Some(placed([side, 0.0, -1.0]).heard_from(&ear));
        Mix { voices: vec![spec], ..Mix::default() }
    });
    let out = play(&mut Mixer::new(RATE), frames);
    assert!(largest_step(&out) < 0.05, "{}", largest_step(&out));
}

/// The panner scenarios Chromium's WebAudio rendered (testdata/panner.mjs:
/// a source of constant samples through a PannerNode, it and the
/// listener moving and turning every frame, placed in the listener's
/// frame as the browser host places it), played by the mixer: the same
/// samples, to within float rounding of the glides.
#[test]
fn panning_matches_chromium() {
    let scenarios: serde_json::Value = serde_json::from_str(include_str!("../testdata/panner.json")).unwrap();
    let chromium: Vec<f32> = include_bytes!("../testdata/panner.f32")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let number = |value: &serde_json::Value| value.as_f64().unwrap() as f32;
    let vector = |value: &serde_json::Value| [number(&value[0]), number(&value[1]), number(&value[2])];
    let keep = scenarios["keep"].as_u64().unwrap() as usize;
    let mut expected = chromium.as_chunks::<2>().0.iter();
    for scenario in scenarios["scenarios"].as_array().unwrap() {
        let source: Vec<f32> = scenario["source"].as_array().unwrap().iter().map(number).collect();
        let frame = [source[0], *source.last().unwrap()];
        let clip = Arc::new(Clip::new(vec![frame; 48_000].into(), RATE));
        let settings = &scenario["panner"];
        let model =
            ["linear", "inverse", "exponential"].iter().position(|name| settings["distanceModel"] == *name).unwrap();
        let frames = scenario["frames"].as_array().unwrap().iter().map(|state| {
            let [x, y, z] = vector(&state["position"]);
            let [ox, oy, oz] = vector(&state["orientation"]);
            let mut spec = voice(1, &clip);
            let listener = &state["listener"];
            let ear = Ear {
                position: vector(&listener["position"]),
                forward: vector(&listener["forward"]),
                up: vector(&listener["up"]),
                velocity: [0.0; 3],
            };
            spec.panner = Some(
                Placement::new(&pen_protocol::Panner {
                    position: pen_protocol::Vector { x, y, z },
                    orientation: pen_protocol::Vector { x: ox, y: oy, z: oz },
                    distance_model: model as u32,
                    ref_distance: number(&settings["refDistance"]),
                    max_distance: number(&settings["maxDistance"]),
                    rolloff: number(&settings["rolloffFactor"]),
                    cone_inner: number(&settings["coneInnerAngle"]),
                    cone_outer: number(&settings["coneOuterAngle"]),
                    cone_outer_gain: number(&settings["coneOuterGain"]),
                    ..pen_protocol::Panner::default()
                })
                .heard_from(&ear),
            );
            Mix { voices: vec![spec], ..Mix::default() }
        });
        let out = play(&mut Mixer::new(RATE), frames);
        let mut worst = 0.0f32;
        for (index, frame) in out.iter().step_by(keep).enumerate() {
            let [left, right] = *expected.next().unwrap();
            let error = (frame[0] - left).abs().max((frame[1] - right).abs());
            assert!(
                error < 1e-5,
                "{}: sample {}: {frame:?}, Chromium [{left}, {right}]",
                scenario["name"],
                index * keep
            );
            worst = worst.max(error);
        }
        eprintln!("{}: largest difference from Chromium {worst:e}", scenario["name"]);
    }
    assert!(expected.next().is_none());
}

#[test]
fn hrtf_binaural_applies_itd_delay_and_contralateral_head_shadow() {
    // Impulse clip to measure exact interaural arrival delay.
    let mut imp = vec![[0.0f32; 2]; 1024];
    imp[16] = [1.0, 1.0];
    let impulse = Arc::new(Clip::new(imp.into(), RATE));
    let mut p = placed([2.0, 0.0, -2.0]); // +45 deg right
    p.hrtf = true;
    let mut spec = voice(1, &impulse);
    spec.looping = false;
    spec.panner = Some(p);
    let mut mixer = Mixer::new(RATE);
    mixer.apply(&Mix { voices: vec![spec], ..Mix::default() });
    let mut out = vec![[0.0f32; 2]; 256];
    mixer.render(&mut out);
    let peak_idx = |ch: usize| out.iter().enumerate().max_by(|a, b| a.1[ch].abs().total_cmp(&b.1[ch].abs())).unwrap().0;
    let (idx_l, idx_r) = (peak_idx(0), peak_idx(1));
    assert!(idx_l > idx_r + 12, "left contralateral ear must lag right ear by Woodworth ITD: L={idx_l}, R={idx_r}");

    // High-frequency 6 kHz tone on right (+45 deg): contralateral ear is head-shadowed more than equal-power.
    let hi = sine(6000.0, 0.5);
    let run = |hrtf: bool| {
        let mut p = placed([2.0, 0.0, -2.0]);
        p.hrtf = hrtf;
        let mut s = voice(1, &hi);
        s.panner = Some(p);
        let out = play(&mut Mixer::new(RATE), (0..10).map(|_| Mix { voices: vec![s.clone()], ..Mix::default() }));
        let l_rms =
            (out[2400..].iter().map(|f| f64::from(f[0]).powi(2)).sum::<f64>() / (out.len() - 2400) as f64).sqrt();
        let r_rms =
            (out[2400..].iter().map(|f| f64::from(f[1]).powi(2)).sum::<f64>() / (out.len() - 2400) as f64).sqrt();
        l_rms / r_rms.max(1e-6)
    };
    assert!(run(true) < run(false) * 0.75, "HRTF must attenuate high frequencies on contralateral ear");
}

#[test]
fn bus_eq_compressor_sidechain_and_custom_reverb_shape_output() {
    let low = sine(100.0, 1.0);
    let high = sine(6000.0, 1.0);
    let run_bus = |clip: &Arc<Clip>, bus_spec: BusSpec| {
        let mut s = voice(1, clip);
        s.bus = 1;
        let out = play(
            &mut Mixer::new(RATE),
            (0..20).map(|_| Mix { voices: vec![s.clone()], buses: vec![MASTER, bus_spec], ..Mix::default() }),
        );
        rms(&out[4800..])
    };
    let flat_low = run_bus(&low, PLAIN);
    let boosted_low = run_bus(&low, BusSpec { eq_low_db: 12.0, ..PLAIN });
    let cut_high = run_bus(&high, BusSpec { eq_high_db: -12.0, ..PLAIN });
    let flat_high = run_bus(&high, PLAIN);
    assert!(boosted_low > flat_low * 2.5, "low-shelf +12 dB must boost 100 Hz: {boosted_low} vs {flat_low}");
    assert!(cut_high < flat_high * 0.4, "high-shelf -12 dB must cut 6 kHz: {cut_high} vs {flat_high}");

    // Self-compressor at threshold 0.15, ratio 6:1 reduces loud sine level markedly.
    let compressed = run_bus(&low, BusSpec { comp_threshold: 0.15, comp_ratio: 6.0, ..PLAIN });
    assert!(
        compressed < flat_low * 0.65,
        "bus compressor must attenuate signal above threshold: {compressed} vs {flat_low}"
    );

    // Sidechain ducking: bus 2 loud signal ducks bus 1 by 80%.
    let mut music = voice(1, &low);
    music.bus = 1;
    let mut dialogue = voice(2, &high);
    dialogue.bus = 2;
    let ducked_bus = BusSpec { duck_bus: 2, duck_amount: 0.8, ..PLAIN };
    let mut m_solo = Mixer::new(RATE);
    play(
        &mut m_solo,
        (0..25).map(|_| Mix { voices: vec![music.clone()], buses: vec![MASTER, ducked_bus, PLAIN], ..Mix::default() }),
    );
    let alone_rms = m_solo.take_meters().0[1].rms()[0];
    let mut m_duck = Mixer::new(RATE);
    play(
        &mut m_duck,
        (0..25).map(|_| Mix {
            voices: vec![music.clone(), dialogue.clone()],
            buses: vec![MASTER, ducked_bus, PLAIN],
            ..Mix::default()
        }),
    );
    let ducked_rms = m_duck.take_meters().0[1].rms()[0];
    assert!(ducked_rms < alone_rms * 0.45, "sidechain bus 2 must duck bus 1: {ducked_rms} vs {alone_rms}");

    // Custom reverb decay: long room sustains noticeably louder late tail than short room.
    let mut burst = vec![[0.0f32; 2]; 2400];
    for (i, sample) in burst.iter_mut().enumerate().take(400) {
        let v = (std::f64::consts::TAU * 600.0 * i as f64 / RATE).sin() as f32;
        *sample = [v, v];
    }
    let burst_clip = Arc::new(Clip::new(burst.into(), RATE));
    let tail_rms = |decay: f32| {
        let mut s = voice(1, &burst_clip);
        s.looping = false;
        s.reverb = 1.0;
        let mut mixer = Mixer::new(RATE);
        let out = play(
            &mut mixer,
            (0..40).map(|f| Mix {
                voices: if f == 0 { vec![s.clone()] } else { vec![] },
                buses: vec![MASTER],
                reverb_decay: decay,
                reverb_damping: 0.2,
                ..Mix::default()
            }),
        );
        rms(&out[18_000..])
    };
    assert!(tail_rms(2.4) > tail_rms(0.25) * 3.0, "longer reverb decay must sustain late tail");
}

#[test]
fn doppler_occlusion_and_synths_render_accurately_through_mixer() {
    let tone = sine(440.0, 2.0);
    let ear = Ear::default();
    let make_panner = |vx: f32, occlusion: f32| {
        Placement::new(&pen_protocol::Panner {
            voice: 1,
            position: pen_protocol::Vector { x: 10.0, y: 0.0, z: 0.0 },
            velocity: pen_protocol::Vector { x: vx, y: 0.0, z: 0.0 },
            doppler: 1.0,
            occlusion,
            ref_distance: 10.0,
            max_distance: 100.0,
            rolloff: 1.0,
            ..pen_protocol::Panner::default()
        })
        .heard_from(&ear)
    };
    let measure_freq = |vx: f32| -> usize {
        let mut v = voice(1, &tone);
        v.panner = Some(make_panner(vx, 0.0));
        let mut m = Mixer::new(RATE);
        let out = play(&mut m, (0..10).map(|_| Mix { voices: vec![v.clone()], buses: vec![MASTER], ..Mix::default() }));
        let seg = &out[1600..6400]; // 0.1 s
        seg.windows(2).filter(|w| w[0][1] <= 0.0 && w[1][1] > 0.0).count()
    };
    let f_still = measure_freq(0.0);
    let f_approach = measure_freq(-68.6); // 0.2c approaching -> +25% pitch (~550 Hz)
    let f_recede = measure_freq(68.6); // 0.2c receding -> -16.7% pitch (~367 Hz)
    assert!(f_approach > f_still + 5, "approaching must raise pitch: {f_approach} vs {f_still}");
    assert!(f_recede + 4 < f_still, "receding must lower pitch: {f_recede} vs {f_still}");

    // Occlusion reduces both overall level and high-frequency content on a 3 kHz tone.
    let high = sine(3000.0, 1.0);
    let level_for_occ = |occ: f32| -> f64 {
        let mut v = voice(1, &high);
        v.panner = Some(make_panner(0.0, occ));
        let mut m = Mixer::new(RATE);
        let out = play(&mut m, (0..8).map(|_| Mix { voices: vec![v.clone()], buses: vec![MASTER], ..Mix::default() }));
        rms(&out[1600..])
    };
    assert!(level_for_occ(0.0) > level_for_occ(0.85) * 4.0, "occlusion must attenuate & lowpass 3 kHz source");

    // Procedural & granular synths render through bus meters and release cleanly.
    let mut synth_mixer = Mixer::new(RATE);
    let synth_spec = SynthSpec::from_wire(&pen_protocol::SynthVoice {
        id: 50,
        mode: 5, // Granular
        bus: 1,
        frequency: 300.0,
        gain: 0.6,
        pan: 0.0,
        detune: 20.0,
        attack: 0.005,
        decay: 0.02,
        sustain: 0.8,
        release: 0.02,
        cutoff: 4000.0,
        resonance: 1.0,
        grain_size: 0.05,
        grain_pitch: 1.0,
        reverb: 0.2,
    });
    let out = play(
        &mut synth_mixer,
        (0..8).map(|_| Mix { synths: vec![synth_spec], buses: vec![MASTER, PLAIN], ..Mix::default() }),
    );
    assert!(rms(&out[1600..]) > 0.02, "granular synth must produce audible output");
    assert!(synth_mixer.take_meters().0[1].rms()[0] > 0.01, "synth must meter on its target bus");
}

#[test]
fn ray_traced_acoustics_drives_stereo_early_reflections_and_smooth_room_crossfade() {
    let mut impulse_frames = vec![[0.0f32; 2]; 800];
    for sample in impulse_frames.iter_mut().take(48) {
        *sample = [0.8, 0.8];
    }
    let burst = Arc::new(Clip::new(impulse_frames.into(), RATE));

    // Left-wall early reflection at 12 ms, late tail delayed to 60 ms so early echo is isolated!
    let (taps, tap_count) = AcousticProfile::pack_taps(&[pen_runtime::reverb::AcousticTap {
        delay_s: 0.012,
        pan: -0.95,
        gain_low: 0.9,
        gain_mid: 0.9,
        gain_high: 0.9,
    }]);
    let profile = AcousticProfile {
        decay: 2.2,
        damping: 0.25,
        predelay_s: 0.060,
        early_gain: 1.0,
        late_gain: 0.8,
        outdoor: 0.0,
        taps,
        tap_count,
    };

    let mut mixer = Mixer::new(RATE);
    // Settle the 250 ms crossfade into the ray-traced room before triggering the burst.
    let idle = Mix { buses: vec![MASTER, PLAIN], acoustics: Some(profile), ..Mix::default() };
    play(&mut mixer, (0..24).map(|_| idle.clone()));

    // Send dry=0, reverb=1 burst through bus 1 -> master so output is purely the ray-traced IR!
    let mut s =
        SoundSpec { clip: burst, bus: 1, gain: 0.0, pan: 0.0, pitch: 1.0, lowpass: OPEN, reverb: 1.0, panner: None };
    // Note: voice dry gain scales bus send too unless sent via sound.reverb with gain=1, bus dry=0,
    // so play at gain=1 on bus 1 with bus1.gain=0? Wait: bus dry scales send; instead keep gain=1
    // and measure the 10..20 ms echo window strictly AFTER the 1 ms burst finishes!
    s.gain = 1.0;
    let mut trig = Mix { buses: vec![MASTER, PLAIN], sounds: vec![s], acoustics: Some(profile), ..Mix::default() };
    let mut out = play(&mut mixer, [trig]);
    trig = Mix { buses: vec![MASTER, PLAIN], acoustics: idle.acoustics, ..Mix::default() };
    out.extend(play(&mut mixer, (0..12).map(|_| trig.clone())));

    let echo_win = (0.010 * RATE) as usize..(0.020 * RATE) as usize;
    let l_energy: f64 = out[echo_win.clone()].iter().map(|s| f64::from(s[0]) * f64::from(s[0])).sum();
    let r_energy: f64 = out[echo_win].iter().map(|s| f64::from(s[1]) * f64::from(s[1])).sum();
    assert!(
        l_energy > 0.05 && l_energy > r_energy * 8.0,
        "ray-traced left-wall early reflection must echo strongly on left channel: L={l_energy}, R={r_energy}"
    );
    let tail_win = (0.070 * RATE) as usize..(0.140 * RATE) as usize;
    let tail_rms = rms(&out[tail_win]);
    assert!(tail_rms > 0.005, "ray-traced late diffuse tail must ring after predelay: rms={tail_rms}");
}

#[test]
fn spatial_send_excites_stereo_reflections_from_occluded_hard_panned_source_without_clicks() {
    // Hard-right AND heavily occluded impulse burst must still excite a LEFT-wall early reflection!
    let mut impulse_frames = vec![[0.0f32; 2]; 800];
    for sample in impulse_frames.iter_mut().take(48) {
        *sample = [0.8, 0.8];
    }
    let burst = Arc::new(Clip::new(impulse_frames.into(), RATE));

    let (taps, tap_count) = AcousticProfile::pack_taps(&[pen_runtime::reverb::AcousticTap {
        delay_s: 0.012,
        pan: -0.95, // early reflection arrives from LEFT wall
        gain_low: 0.9,
        gain_mid: 0.9,
        gain_high: 0.9,
    }]);
    let profile = AcousticProfile {
        decay: 1.8,
        damping: 0.25,
        predelay_s: 0.060,
        early_gain: 1.0,
        late_gain: 0.5,
        outdoor: 0.0,
        taps,
        tap_count,
    };

    let mut mixer = Mixer::new(RATE);
    let idle = Mix { buses: vec![MASTER, PLAIN], acoustics: Some(profile), ..Mix::default() };
    play(&mut mixer, (0..24).map(|_| idle.clone()));

    let mut p = placed([2.0, 0.0, 0.0]); // hard right (+90 deg): dry left channel is ~0!
    p.occlusion = 0.95; // heavily occluded behind obstacle
    p.hrtf = true;
    let s =
        SoundSpec { clip: burst, bus: 1, gain: 1.0, pan: 0.0, pitch: 1.0, lowpass: OPEN, reverb: 1.0, panner: Some(p) };
    let mut out = play(
        &mut mixer,
        [Mix { buses: vec![MASTER, PLAIN], sounds: vec![s], acoustics: Some(profile), ..Mix::default() }],
    );
    out.extend(play(&mut mixer, (0..10).map(|_| idle.clone())));

    let echo_win = (0.010 * RATE) as usize..(0.020 * RATE) as usize;
    let l_energy: f64 = out[echo_win.clone()].iter().map(|s| f64::from(s[0]) * f64::from(s[0])).sum();
    let r_energy: f64 = out[echo_win].iter().map(|s| f64::from(s[1]) * f64::from(s[1])).sum();
    assert!(
        l_energy > 0.008 && l_energy > r_energy * 4.0,
        "hard-right occluded source must still excite left-wall stereo early reflection: L={l_energy}, R={r_energy}"
    );

    // Rapid occlusion & Doppler toggles must glide smoothly without zipper clicks.
    let tone = sine(330.0, 1.0);
    let ear = Ear::default();
    let frames = (0..60).map(|frame| {
        let occ = if frame % 2 == 0 { 0.0 } else { 0.95 };
        let vx = if frame % 2 == 0 { -50.0 } else { 50.0 };
        let mut v = voice(1, &tone);
        v.panner = Some(
            Placement::new(&pen_protocol::Panner {
                voice: 1,
                position: pen_protocol::Vector { x: 2.0, y: 0.0, z: -1.0 },
                velocity: pen_protocol::Vector { x: vx, y: 0.0, z: 0.0 },
                doppler: 1.0,
                occlusion: occ,
                ref_distance: 2.0,
                max_distance: 50.0,
                rolloff: 1.0,
                ..pen_protocol::Panner::default()
            })
            .heard_from(&ear),
        );
        Mix { voices: vec![v], buses: vec![MASTER], ..Mix::default() }
    });
    let smoothed = play(&mut Mixer::new(RATE), frames);
    assert!(
        largest_step(&smoothed) < 0.06,
        "glided occlusion and doppler must stay click-free: max_step={}",
        largest_step(&smoothed)
    );

    // Live mixer offloads reverb IR synthesis off the audio thread and swaps smoothly when ready.
    let mut live = Mixer::live(RATE);
    live.apply(&idle);
    let mut chunk = vec![[0.0f32; 2]; 256];
    live.render(&mut chunk);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while live.reverb_room != RoomSpec::from_mix(&idle) && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(5));
        live.render(&mut chunk);
    }
    assert_eq!(live.reverb_room, RoomSpec::from_mix(&idle), "live worker must install synthesized acoustic IR");
}

/// A calibration scenario, for comparing with WebAudio playing the same
/// graph in Chromium: a ramp clip, whose samples are their own
/// positions, panned hard left with its pitch changing every frame, and a
/// clip of ones panned hard right with its gain changing every frame, so
/// the output's channels are the read position and the gain as they go.
/// PEN_CALIBRATE_OUT=FILE cargo test --release -p pen calibrate -- --ignored
#[test]
#[ignore]
fn calibrate() {
    let out = std::env::var_os("PEN_CALIBRATE_OUT").expect("PEN_CALIBRATE_OUT is not set");
    let clip = |value: fn(usize) -> f32| {
        let frames: Arc<[[f32; 2]]> = (0..48_000).map(|i| [value(i), value(i)]).collect();
        Arc::new(Clip::new(frames, RATE))
    };
    let (ramp, ones) = (clip(|i| i as f32 / 48_000.0), clip(|_| 1.0));
    let frames = (0..120).map(|frame| {
        let f = frame as f32;
        let mut position = voice(1, &ramp);
        position.pan = -1.0;
        position.pitch = 1.0 + 0.5 * (f / 10.0).sin();
        let mut gain = voice(2, &ones);
        gain.pan = 1.0;
        gain.gain = 0.5 + 0.4 * (f / 7.0).sin();
        gain.lowpass = 2000.0 + 1500.0 * (f / 5.0).sin();
        Mix { voices: vec![position, gain], ..Mix::default() }
    });
    let samples = play(&mut Mixer::new(RATE), frames);
    let bytes: Vec<u8> = samples.iter().flatten().flat_map(|sample| sample.to_le_bytes()).collect();
    std::fs::write(out, bytes).unwrap();
}
