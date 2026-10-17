//! Clips decoded as they play: a voice of a long clip
//! (`pen_runtime::clip::STREAM_SECONDS`) reads its frames from a ring that a
//! streamer fills a little ahead of it, so that only the clip's encoded
//! bytes stay in memory.
//!
//! The streamer decodes the clip from its start, resamples it to the
//! mixer's rate exactly as the whole clip would be (`resample.rs`, each pass
//! round a looping clip on its own), and writes the frames in the order the
//! voice plays them: from where it starts to the end, then round again
//! while it loops. The voice reads them by that order's index, so a stream
//! plays sample for sample what the clip decoded whole would. A voice that
//! catches up with its stream (it ran dry) plays silence and keeps time,
//! and the streamer skips what it missed; one whose stream has not begun
//! waits for it, as the browser waits for a clip to decode.
//!
//! Live, a thread of its own streams (`Streams::live`), waking every
//! couple of milliseconds; offline (recordings, tests), the mixer streams
//! on its own thread before each quantum (`Streams::offline`), so nothing
//! runs dry. The voice and the streamer share a slot of a fixed pool: the
//! ring is atomics (each frame two f32s in a u64), and nothing is allocated
//! or freed on the audio thread.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "frames packed in u64s, places in the ring, and taps"
)]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::time::Duration;

use super::resample::{Resampler, TAPS};
use super::source::{Encoded, Source};

/// Frames a stream's ring holds: 1.4 s at 48 kHz.
const CAPACITY: u64 = 1 << 16;
/// Streams playing at once; a voice past them plays silence. With the
/// rings, 4 MB.
pub const STREAMS: usize = 8;
/// How long the live streamer sleeps when every ring is full.
const NAP: Duration = Duration::from_millis(2);

/// A slot's state: free, a voice's, or let go by its voice and not yet
/// cleared by the streamer.
const FREE: u8 = 0;
const CLAIMED: u8 = 1;
const RELEASED: u8 = 2;

fn pack([left, right]: [f32; 2]) -> u64 {
    u64::from(left.to_bits()) | u64::from(right.to_bits()) << 32
}

fn unpack(bits: u64) -> [f32; 2] {
    [f32::from_bits((bits & 0xffff_ffff) as u32), f32::from_bits((bits >> 32) as u32)]
}

/// What a voice and the streamer share: the ring, how far each has got in
/// it, and whether the voice loops.
pub struct Slot {
    state: AtomicU8,
    frames: Box<[AtomicU64]>,
    /// Frames written, in the order the voice plays them.
    written: AtomicU64,
    /// Frames the voice is done with: it plays on from here.
    read: AtomicU64,
    looping: AtomicBool,
}

impl Slot {
    fn new() -> Slot {
        Slot {
            state: AtomicU8::new(FREE),
            frames: (0..CAPACITY).map(|_| AtomicU64::new(0)).collect(),
            written: AtomicU64::new(0),
            read: AtomicU64::new(0),
            looping: AtomicBool::new(false),
        }
    }
}

/// A stream's start, for the streamer: the slot, the clip, the mixer's
/// rate and the frame (at that rate) the voice starts from.
struct Request {
    slot: usize,
    clip: Arc<Encoded>,
    rate: f64,
    first: usize,
}

/// A voice's stream: the frames it plays, by their place in its order.
pub struct Reader {
    slot: Arc<Slot>,
    underruns: Arc<AtomicU64>,
    /// Frames written when the quantum began.
    written: u64,
    /// The quantum ran dry.
    dry: bool,
}

impl Reader {
    /// Whether the stream has begun.
    pub fn begun(&self) -> bool {
        self.slot.written.load(Ordering::Acquire) > 0
    }

    /// A quantum begins: what has been written so far is what it plays.
    pub fn begin(&mut self, looping: bool) {
        self.slot.looping.store(looping, Ordering::Relaxed);
        self.written = self.slot.written.load(Ordering::Acquire);
        self.dry = false;
    }

    /// Frame `index` of the voice's order, or silence if it is not written
    /// yet.
    pub fn frame(&mut self, index: u64) -> [f32; 2] {
        if index < self.written {
            unpack(self.slot.frames[(index % CAPACITY) as usize].load(Ordering::Relaxed))
        } else {
            self.dry = true;
            [0.0; 2]
        }
    }

    /// The quantum is over: the voice plays on from frame `index`.
    pub fn end(&mut self, index: u64) {
        self.slot.read.store(index, Ordering::Release);
        if self.dry {
            self.underruns.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.slot.state.store(RELEASED, Ordering::Release);
    }
}

/// The mixer's side: the pool of slots, and where stream starts go.
pub struct Streams {
    slots: Vec<Arc<Slot>>,
    requests: SyncSender<Request>,
    /// Offline, the streamer itself, run before each quantum.
    inline: Option<Streamer>,
    underruns: Arc<AtomicU64>,
}

impl Streams {
    fn pair() -> (Streams, Streamer) {
        let slots: Vec<Arc<Slot>> = (0..STREAMS).map(|_| Arc::new(Slot::new())).collect();
        let (requests, received) = sync_channel(STREAMS * 2);
        let streamer =
            Streamer { slots: slots.clone(), requests: received, decoding: (0..STREAMS).map(|_| None).collect() };
        (Streams { slots, requests, inline: None, underruns: Arc::new(AtomicU64::new(0)) }, streamer)
    }

    /// Streams the mixer fills itself, before each quantum: nothing runs
    /// dry, however fast it renders.
    pub fn offline() -> Streams {
        let (mut streams, streamer) = Streams::pair();
        streams.inline = Some(streamer);
        streams
    }

    /// Streams a thread of their own fills, until these are dropped.
    pub fn live() -> Streams {
        let (streams, mut streamer) = Streams::pair();
        let spawned = std::thread::Builder::new().name("pen-audio-stream".into()).spawn(move || {
            while let Some(worked) = streamer.pump() {
                if !worked {
                    std::thread::sleep(NAP);
                }
            }
        });
        if let Err(error) = spawned {
            eprintln!("audio streams are off: {error}");
        }
        streams
    }

    /// A stream of `clip` at `rate`, from frame `first` of it at that rate;
    /// none while every slot is taken.
    pub fn open(&mut self, clip: &Arc<Encoded>, rate: f64, first: usize) -> Option<Reader> {
        let index = self.slots.iter().position(|slot| {
            slot.state.compare_exchange(FREE, CLAIMED, Ordering::Acquire, Ordering::Relaxed).is_ok()
        })?;
        let slot = self.slots[index].clone();
        slot.written.store(0, Ordering::Relaxed);
        slot.read.store(0, Ordering::Relaxed);
        let reader = Reader { slot, underruns: self.underruns.clone(), written: 0, dry: false };
        // A full queue lets the slot go again: the voice plays silence.
        self.requests.try_send(Request { slot: index, clip: clip.clone(), rate, first }).ok()?;
        Some(reader)
    }

    /// Offline, the streams filled for the next quantum.
    pub fn pump(&mut self) {
        if let Some(streamer) = &mut self.inline {
            streamer.pump();
        }
    }

    /// How often a voice ran dry so far.
    pub fn underruns(&self) -> u64 {
        self.underruns.load(Ordering::Relaxed)
    }
}

/// A stream being decoded: the pass round the clip it is on.
struct Decoding {
    clip: Arc<Encoded>,
    source: Box<dyn Source>,
    /// None when the clip is at the mixer's rate.
    resampler: Option<Resampler>,
    /// Decoded frames of this pass from `input_start` on.
    input: Vec<[f32; 2]>,
    input_start: usize,
    /// The source has no more frames this pass.
    ended: bool,
    /// The next frame of this pass, at the mixer's rate, and how many it has.
    output: usize,
    length: usize,
    /// Frames of the first pass before the voice's start, left out.
    skip: usize,
    /// The place in the voice's order of the next frame written.
    next: u64,
}

impl Decoding {
    fn new(request: Request) -> anyhow::Result<Decoding> {
        let Request { clip, rate, first, .. } = request;
        let resampler = ((clip.rate - rate).abs() >= 0.5).then(|| Resampler::new(clip.rate, rate));
        let length = resampler.as_ref().map_or(clip.frames, |resampler| resampler.length(clip.frames));
        Ok(Decoding {
            source: clip.source()?,
            clip,
            resampler,
            input: Vec::new(),
            input_start: 0,
            ended: false,
            output: 0,
            length,
            skip: first.min(length),
            next: 0,
        })
    }

    /// Input frame `index` of this pass, decoding as far as it; silence
    /// past the clip's end.
    fn input(&mut self, index: usize) -> anyhow::Result<[f32; 2]> {
        if index >= self.clip.frames {
            return Ok([0.0; 2]);
        }
        while !self.ended && index >= self.input_start + self.input.len() {
            self.ended = !self.source.next(&mut self.input)?;
        }
        Ok(self.input.get(index - self.input_start).copied().unwrap_or([0.0; 2]))
    }

    /// The next frame of this pass.
    fn frame(&mut self) -> anyhow::Result<[f32; 2]> {
        let j = self.output;
        let Some((first, keep)) = self.resampler.as_ref().map(|resampler| (resampler.first(j), resampler.first(j + 1)))
        else {
            let frame = self.input(j)?;
            self.forget(j + 1);
            return Ok(frame);
        };
        if let Ok(last) = usize::try_from(first + TAPS as i64 - 1) {
            self.input(last)?;
        }
        let (input, start, frames) = (&self.input, self.input_start, self.clip.frames);
        let at = |side: usize| {
            move |index: i64| match usize::try_from(index) {
                Ok(index) if index < frames && index >= start => {
                    input.get(index - start).map_or(0.0, |frame| frame[side])
                }
                _ => 0.0,
            }
        };
        let resampler = self.resampler.as_ref().expect("a resampler");
        let left = resampler.sample(j, at(0));
        let right = if self.clip.channels == 1 { left } else { resampler.sample(j, at(1)) };
        self.forget(usize::try_from(keep).unwrap_or(0));
        Ok([left, right])
    }

    /// Input frames before `index` are needed no more.
    fn forget(&mut self, index: usize) {
        if index > self.input_start + 4096 {
            let drop = (index - self.input_start).min(self.input.len());
            self.input.drain(..drop);
            self.input_start += drop;
        }
    }

    /// Round the clip again, from its start.
    fn again(&mut self) -> anyhow::Result<()> {
        self.source = self.clip.source()?;
        self.input.clear();
        self.input_start = 0;
        self.ended = false;
        self.output = 0;
        Ok(())
    }

    /// Fills `slot`'s ring as far as it has room; whether it wrote any.
    fn fill(&mut self, slot: &Slot) -> anyhow::Result<bool> {
        let mut wrote = false;
        loop {
            let read = slot.read.load(Ordering::Acquire);
            if self.next >= read + CAPACITY {
                return Ok(wrote);
            }
            if self.output == self.length {
                if !slot.looping.load(Ordering::Relaxed) || self.length == 0 {
                    return Ok(wrote);
                }
                self.again()?;
            }
            let frame = self.frame()?;
            self.output += 1;
            if self.skip > 0 {
                self.skip -= 1;
                continue;
            }
            // The voice ran dry and went on: what it missed is not written.
            if self.next >= read {
                slot.frames[(self.next % CAPACITY) as usize].store(pack(frame), Ordering::Relaxed);
            }
            self.next += 1;
            slot.written.store(self.next, Ordering::Release);
            wrote = true;
        }
    }
}

/// The decoding side: each claimed slot's stream.
struct Streamer {
    slots: Vec<Arc<Slot>>,
    requests: Receiver<Request>,
    decoding: Vec<Option<Decoding>>,
}

impl Streamer {
    /// Starts the streams asked for, clears the slots let go and fills the
    /// rest; whether it did anything, or none once the mixer is gone.
    fn pump(&mut self) -> Option<bool> {
        let mut worked = false;
        loop {
            match self.requests.try_recv() {
                Ok(request) => {
                    let slot = request.slot;
                    let rate = request.rate;
                    self.decoding[slot] = match Decoding::new(request) {
                        Ok(decoding) => Some(decoding),
                        Err(error) => {
                            eprintln!("streaming a clip at {rate} Hz: {error:#}");
                            None
                        }
                    };
                    worked = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return None,
            }
        }
        for (slot, decoding) in self.slots.iter().zip(&mut self.decoding) {
            match slot.state.load(Ordering::Acquire) {
                RELEASED => {
                    *decoding = None;
                    slot.state.store(FREE, Ordering::Release);
                    worked = true;
                }
                CLAIMED => {
                    if let Some(stream) = decoding {
                        match stream.fill(slot) {
                            Ok(wrote) => worked |= wrote,
                            Err(error) => {
                                eprintln!("streaming a clip: {error:#}");
                                *decoding = None;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Some(worked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mono 48 kHz clip whose samples count up, streamed as written.
    fn counting(frames: usize) -> Arc<Encoded> {
        let samples: Vec<i16> = (0..frames).map(|i| i16::try_from(i % 30_000).unwrap()).collect();
        let size = u32::try_from(samples.len() * 2).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + size).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        for value in [16u32, 1 | 1 << 16, 48_000, 96_000, 2 | 16 << 16] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend(samples.iter().flat_map(|sample| sample.to_le_bytes()));
        Arc::new(Encoded::new(bytes.into()).unwrap())
    }

    fn value(frame: [f32; 2]) -> i64 {
        (f64::from(frame[0]) * 32_768.0).round() as i64
    }

    /// A stream begins once the streamer has seen it, from the voice's
    /// first frame, and stays ahead of the voice by a ring's length.
    #[test]
    fn streams_begin_where_the_voice_starts_and_stay_ahead() {
        let (mut streams, mut streamer) = Streams::pair();
        let mut reader = streams.open(&counting(200_000), 48_000.0, 1000).unwrap();
        assert!(!reader.begun());
        streamer.pump();
        assert!(reader.begun());
        reader.begin(false);
        assert_eq!((value(reader.frame(0)), value(reader.frame(9))), (1000, 1009));
        assert_eq!(reader.written, CAPACITY);
        // Played on, the streamer writes on behind it.
        reader.end(50_000);
        streamer.pump();
        reader.begin(false);
        assert_eq!(reader.written, 50_000 + CAPACITY);
        assert_eq!(value(reader.frame(50_000 + CAPACITY - 1)), i64::from((1000 + 50_000 + 65_535) % 30_000));
    }

    /// A voice that runs dry plays silence, counts it, and goes on; the
    /// streamer writes from where it went on to.
    #[test]
    fn a_voice_that_runs_dry_keeps_time() {
        let (mut streams, mut streamer) = Streams::pair();
        let mut reader = streams.open(&counting(400_000), 48_000.0, 0).unwrap();
        streamer.pump();
        reader.begin(false);
        assert_eq!(reader.frame(CAPACITY + 5), [0.0; 2]);
        reader.end(CAPACITY + 100_000);
        assert_eq!(streams.underruns(), 1);
        streamer.pump();
        reader.begin(false);
        assert_eq!(value(reader.frame(CAPACITY + 100_000)), i64::from((65_536 + 100_000) % 30_000));
    }

    /// A looping stream goes round from the clip's start; one that does not
    /// stops at its end.
    #[test]
    fn streams_go_round_while_looping() {
        let (mut streams, mut streamer) = Streams::pair();
        let mut reader = streams.open(&counting(1000), 48_000.0, 900).unwrap();
        reader.begin(true);
        streamer.pump();
        reader.begin(true);
        assert_eq!((value(reader.frame(99)), value(reader.frame(100)), value(reader.frame(1100))), (999, 0, 0));
        let mut once = streams.open(&counting(1000), 48_000.0, 900).unwrap();
        once.begin(false);
        streamer.pump();
        once.begin(false);
        assert_eq!(once.written, 100);
    }

    /// Slots are a fixed pool: past it no stream opens, and a voice letting
    /// go of one frees it for the next once the streamer has cleared it.
    #[test]
    fn the_pool_frees_what_voices_let_go() {
        let (mut streams, mut streamer) = Streams::pair();
        let clip = counting(10_000);
        let mut readers: Vec<Reader> = (0..STREAMS).map(|_| streams.open(&clip, 48_000.0, 0).unwrap()).collect();
        assert!(streams.open(&clip, 48_000.0, 0).is_none());
        streamer.pump();
        drop(readers.pop());
        assert!(streams.open(&clip, 48_000.0, 0).is_none(), "not until the streamer clears it");
        streamer.pump();
        let mut again = streams.open(&clip, 48_000.0, 500).unwrap();
        streamer.pump();
        again.begin(false);
        assert_eq!(value(again.frame(0)), 500);
        // The mixer gone, the streamer stops.
        drop(streams);
        assert_eq!(streamer.pump(), None);
    }
}
