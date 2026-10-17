// GPU timestamps around a frame's passes, where the device has them
// (`TIMESTAMP_QUERY`) and something wants them (the dev tools' diagnostics,
// or a frame budget): each timed pass writes one when it begins and one when
// it ends. Timestamps between passes keep a GPU from overlapping them, which
// cost gdev's frames about 0.3 ms on an RTX 2070 SUPER, so frames nobody
// times write none. The frame's are resolved into a buffer that is read back once the
// GPU is done with it, a few frames later (readback.rs), so measuring never
// waits on the GPU; the latest frame read back is what `RenderStats::gpu`
// shows.
use std::cell::RefCell;

use super::readback::Readbacks;
use crate::GpuTimes;

/// Which part of a frame a timed pass belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Span {
    /// The sun's shadow maps.
    Shadow,
    /// Instances culled on the GPU.
    Culling,
    /// The scene's passes: opaque surfaces, decals, translucent ones and overlays.
    Scene,
    /// The passes after the scene: occlusion, sky and fog, bloom, present.
    Post,
    /// Sprites and insets over the output.
    Overlay,
}

// Timestamps a frame can write: a begin and an end for each of 32 passes.
// Passes past them go untimed.
const QUERIES: u32 = 64;

// Frames whose timestamps may be on their way back at once.
const READBACKS: usize = 4;

struct Queries {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    /// Each readback's timed passes' spans, in query order.
    readbacks: Readbacks<Vec<Span>>,
    /// Nanoseconds a tick.
    period: f64,
}

pub(crate) struct GpuTimer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Made on the first timed frame; none without timestamps.
    queries: Option<Queries>,
    /// The readback this frame's timestamps go to, while it is timed.
    frame: Option<usize>,
    /// This frame's timed passes' spans, in query order.
    passes: RefCell<Vec<Span>>,
    latest: Option<GpuTimes>,
    /// Whether `latest` came back since `fresh` last said.
    fresh: bool,
}

impl GpuTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            queries: None,
            frame: None,
            passes: RefCell::new(Vec::new()),
            latest: None,
            fresh: false,
        }
    }

    /// Whether the device has timestamps to time frames with.
    pub fn timed(&self) -> bool {
        self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    /// The whole of the last frame whose timestamps have come back, in
    /// milliseconds, if they came back since this was last asked.
    pub fn fresh(&mut self) -> Option<f32> {
        std::mem::take(&mut self.fresh).then_some(self.latest?.total)
    }

    /// The last frame whose timestamps have come back.
    pub fn latest(&self) -> Option<GpuTimes> {
        self.latest
    }

    /// Takes in the timestamps that have come back, then times the frame
    /// about to be encoded, if `wanted` and a readback is free.
    pub fn begin(&mut self, wanted: bool) {
        self.frame = None;
        self.passes.borrow_mut().clear();
        if !self.timed() || (!wanted && self.queries.is_none()) {
            return;
        }
        let device = &self.device;
        let queue = &self.queue;
        let queries = self.queries.get_or_insert_with(|| Queries {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERIES,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamps"),
                size: u64::from(QUERIES) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readbacks: Readbacks::new("timestamps read back", READBACKS, device),
            period: f64::from(queue.get_timestamp_period()),
        });
        let (period, latest, fresh) = (queries.period, &mut self.latest, &mut self.fresh);
        queries.readbacks.take(|bytes, spans| {
            let ticks: Vec<u64> = bytes.as_chunks::<8>().0.iter().map(|&tick| u64::from_le_bytes(tick)).collect();
            *latest = Some(times(spans, &ticks, period));
            *fresh = true;
        });
        if wanted {
            self.frame = queries.readbacks.free(device, u64::from(QUERIES) * 8);
        } else {
            self.latest = None;
        }
    }

    /// The timestamps a pass of `span` writes, if this frame is timed and
    /// has queries left.
    pub fn pass(&self, span: Span) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let (set, first) = self.next(span)?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: set,
            beginning_of_pass_write_index: Some(first),
            end_of_pass_write_index: Some(first + 1),
        })
    }

    /// The timestamps a compute pass of `span` writes, likewise.
    pub fn compute(&self, span: Span) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (set, first) = self.next(span)?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: set,
            beginning_of_pass_write_index: Some(first),
            end_of_pass_write_index: Some(first + 1),
        })
    }

    fn next(&self, span: Span) -> Option<(&wgpu::QuerySet, u32)> {
        self.frame?;
        let queries = self.queries.as_ref()?;
        let mut passes = self.passes.borrow_mut();
        let first = u32::try_from(passes.len() * 2).ok().filter(|&first| first + 2 <= QUERIES)?;
        passes.push(span);
        Some((&queries.set, first))
    }

    /// Copies the frame's timestamps out to its readback; `submitted`
    /// follows once the encoder is submitted.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let (Some(frame), Some(queries)) = (self.frame, &self.queries) else { return };
        let count = u32::try_from(self.passes.borrow().len() * 2).expect("at most QUERIES");
        if count == 0 {
            return;
        }
        encoder.resolve_query_set(&queries.set, 0..count, &queries.resolve, 0);
        let readback = queries.readbacks.buffer(frame);
        encoder.copy_buffer_to_buffer(&queries.resolve, 0, readback, 0, u64::from(count) * 8);
    }

    /// Reads the frame's timestamps back once the GPU is done with them.
    pub fn submitted(&mut self) {
        let (Some(frame), Some(queries)) = (self.frame.take(), &mut self.queries) else { return };
        let spans = std::mem::take(&mut *self.passes.borrow_mut());
        if !spans.is_empty() {
            queries.readbacks.submitted(frame, spans.len() as u64 * 16, spans);
        }
    }
}

/// Each span's milliseconds from its passes' begin and end ticks, and the
/// whole frame's from the first begin to the last end. A pass whose end
/// reads before its begin (a backend that does not keep them in order)
/// counts nothing.
fn times(spans: &[Span], ticks: &[u64], period: f64) -> GpuTimes {
    let mut times = GpuTimes::default();
    let milliseconds = |ticks: u64| {
        #[allow(clippy::cast_possible_truncation, reason = "milliseconds to show")]
        let shown = (ticks as f64 * period / 1e6) as f32;
        shown
    };
    let (mut first, mut last) = (u64::MAX, 0);
    for (span, pair) in spans.iter().zip(ticks.as_chunks::<2>().0) {
        let [begin, end] = *pair;
        first = first.min(begin);
        last = last.max(end);
        let taken = milliseconds(end.saturating_sub(begin));
        match span {
            Span::Shadow => times.shadow += taken,
            Span::Culling => times.culling += taken,
            Span::Scene => times.scene += taken,
            Span::Post => times.post += taken,
            Span::Overlay => times.overlay += taken,
        }
    }
    times.total = milliseconds(last.saturating_sub(first));
    times
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_add_their_passes_and_the_frame_spans_them_all() {
        // Ticks of 2 microseconds: shadow 100..300, scene 400..1400 and
        // 1500..1600, post 1600..2100, and a pass read out of order.
        let spans = [Span::Shadow, Span::Scene, Span::Scene, Span::Post, Span::Overlay];
        let ticks = [100, 300, 400, 1400, 1500, 1600, 1600, 2100, 900, 800];
        let times = times(&spans, &ticks, 2_000.0);
        assert_eq!(times.shadow, 0.4);
        assert_eq!(times.scene, 2.2);
        assert_eq!(times.post, 1.0);
        assert_eq!(times.overlay, 0.0);
        assert_eq!(times.culling, 0.0);
        assert_eq!(times.total, 4.0);
    }
}
