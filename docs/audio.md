# Audio

pen plays the game's sound twice over: natively, `crates/pen/src/audio/`
emulates the WebAudio graph sample for sample on cpal's output; in the
browser, `crates/pen-web/src/audio.rs` builds that graph from real WebAudio
nodes. README's "Sound" describes the graph. This page covers what sits on
top of it: sounds placed in the world, streamed music, the output's latency,
the dev tools' meters, loudness and captions.

## Sounds placed in the world

A render packet may place voices and sounds in the world (`panners`) and say
where they are heard from (`listener`). A placed voice or sound pans through
WebAudio's PannerNode with equal-power panning instead of its
StereoPannerNode: by its azimuth around the listener, quieted by its distance
model (`linear`, `inverse` or `exponential`, as `DISTANCE_MODEL_*`;
`ref_distance`, `max_distance`, `rolloff`) and its cone (`orientation`,
`cone_inner`, `cone_outer`, `cone_outer_gain`). Its own `pan` is not used;
its gain, lowpass, fades and reverb send are. With no `listener`, the camera
listens: `view`'s eye, facing where it looks.

`pen_runtime::panner` holds the arithmetic both hosts share: Chrome's
azimuth (`PannerHandler::CalculateAzimuthElevation`), equal-power gains, the
distance models and the cone, and the sanitizing that keeps every value one a
PannerNode setter accepts. Positions and orientations glide toward each
frame's values as gains do (AUDIO_GLIDE_MS); the other settings change at once.
A voice that gains or loses its panner starts afresh, as one whose bus
changed.

Both hosts place panners in the listener's own frame and leave WebAudio's
AudioListener where it starts. A still scene sounds the same either way, but
Chrome runs AudioListener automation at twice the rate it should: a
`setTargetAtTime` on `listener.forwardX` reaches, after n samples, where the
same glide on a PannerNode reaches after 2n. Gliding only PannerNode values
keeps the hosts alike while things move.

`panning_matches_chromium` (mixer.rs) plays three scenarios Chromium rendered
offline (`crates/pen/src/audio/testdata/panner.mjs`: a source of constant
samples, so the output is the panner's gains, moving and turning with the
listener every frame) and matches them to within 1e-6 (it allows 1e-5). To
render them again:

```
PLAYWRIGHT_MODULE=$PWD/target/browser-tools/node_modules/playwright-core/index.mjs \
  node crates/pen/src/audio/testdata/panner.mjs
```

HRTF panning (`panningModel: "HRTF"`) is left for later: Chrome's uses its
own measured responses, which the native mixer would need to match.

ink's `pen/host` sends no panners yet: gdev places its sounds itself
(`ink/sound`'s `hear`). Sending `audio.Sound`'s and `audio.Voice`'s
`position` as panners instead would hand that to the host.

## Streamed clips

A clip longer than `pen_runtime::clip::STREAM_SECONDS` (12 s, by its header:
a WAV's data size, an Ogg Opus stream's last granule position) streams on
both hosts: only its bytes stay in memory. Natively, a voice playing one
reads a ring that a streamer thread keeps 1.4 s ahead of it
(`audio/stream.rs`): it decodes the clip a packet at a time
(`audio/source.rs`, `opus.rs`), resamples it as the whole clip would be
resampled, and writes the frames in the order the voice plays them, round
again while it loops. A streamed voice plays sample for sample what the clip
decoded whole plays (`streams_play_what_whole_clips_play`), with any start,
pitch and loop. Eight streams play at once (4 MB of rings); a voice that
catches up with its stream plays silence, keeps time and is counted
(`underruns` in the dev tools' audio answer). Recordings and tests stream on
the mixer's own thread before each quantum, so nothing runs dry however fast
they render.

In the browser, a long clip's bytes become a Blob URL that an
`HTMLAudioElement` streams into the same nodes through a
MediaElementAudioSourceNode, its pitch the element's `playbackRate`
(`preservesPitch` off). It starts when the element can, not at a sample.

## Latency and the device's buffer

`--audio-buffer N` (or `PEN_AUDIO_BUFFER=N`) asks the device for N samples a
callback, within the sizes it takes; without it the system chooses. Smaller
plays sooner after a frame, at more risk of gaps. Each callback tells the
mixer when its first sample will play; the output's latency is that, plus
half a callback and half a render quantum, the average wait of a frame's mix
for the callback and the quantum it lands in, averaged over callbacks. In the
browser, `?audio-latency=interactive|balanced|playback|SECONDS` is the
AudioContext's latency hint, and the latency is its `baseLatency` plus
`outputLatency`.

## Meters, mute and solo

The dev tools service's `audio` operation answers DEVTOOLS_AUDIO's numbers
(latency, rate, buffer, voices, streams, underruns, buses) and then each
bus's DEVTOOLS_BUS numbers: peak and RMS on each side since the last answer
(in the browser, over the last 2048 samples an AnalyserNode holds), and its
state. `mute` and `solo` (bus `a`, on while `b` is 1) are the developer's:
their buses' gains glide to 0, never in the game's packets nor a recording.
While any bus is soloed, only soloed buses are heard, with the master's own
voices; muting the master silences everything.

## Loudness

`pen loudness FILE...` measures WAV and Ogg Opus clips as EBU R128 does
(ITU-R BS.1770-4: K-weighted, 400 ms blocks every 100 ms, gated at -70 LUFS
and 10 LU below the rest), with their true peak (oversampled four times) and
length. `--normalize` brings each to its target (`--lufs`, default -23 LUFS),
less if its true peak would pass the ceiling (`--ceiling`, default -1 dBTP):
a WAV's samples are scaled where they are, an Opus file's head output gain
(RFC 7845) changes instead, which libopus here and Chromium's decoder both
apply. A clip within 0.1 LU of its target is left as it is. The
measurements agree with ffmpeg's `ebur128` filter to 0.1 LU and 0.1 dB. `--targets FILE` reads targets as JSON:

```json
{ "target": -23, "peak": -1, "sounds": { "music.ogg": -20 } }
```

gdev's `just sounds` normalizes its synthesized sounds to the targets in
`models/sounds/loudness.json`, which hold each sound where it sat when the
targets were set, so that changing a sound's synthesis does not change the
mix.

## Captions

A sound may carry a caption: the assets' `captions.json` maps a sound's
catalog key to its text, or to its text and how many seconds it shows
(default 3):

```json
{ "kaching.wav": { "text": "[Till rings]", "seconds": 2 }, "wind.wav": "[Wind gusts]" }
```

As frames play sounds, `pen_runtime::captions::Track` keeps the captions
showing: a sound's from when it plays, a voice's from when it starts (a new
id, or another clip), the same caption again showing longer rather than
twice. Both hosts show them to assistive technology as a polite live region
of their own (natively a Status node after the game's, in the browser a
`role="status"` element), and answer the captions service:
`CAPTIONS_OP_TEXT` gives clip `a`'s caption and seconds, `CAPTIONS_OP_ACTIVE`
the captions showing, newest first, a line each. Packages ship
`captions.json` with the catalog.
