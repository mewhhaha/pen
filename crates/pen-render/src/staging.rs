//! Frame-scoped GPU staging ring buffer for encoder-ordered uniform and small-buffer uploads.
//!
//! Eliminates per-frame `device.create_buffer(mapped_at_creation: true)` allocations across
//! post-processing, ray tracing, SVGF, ReSTIR, DDGI, atmosphere, meshlet, and particle passes
//! while preserving exact intra-command-buffer ordering when the same destination buffer is
//! updated multiple times within a single `CommandEncoder`.
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

const INITIAL_RING_BYTES: u64 = 256 * 1024;
const COPY_ALIGN: u64 = 16;

struct RingState {
    queue: wgpu::Queue,
    buffer: Option<wgpu::Buffer>,
    cursor: u64,
    peak: u64,
}

/// Reusable GPU upload ring owned by the renderer.
#[derive(Clone)]
pub(crate) struct FrameStaging {
    inner: Arc<Mutex<RingState>>,
}

thread_local! {
    static ACTIVE_STAGING: RefCell<Option<FrameStaging>> = const { RefCell::new(None) };
}

struct ActiveGuard(Option<FrameStaging>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE_STAGING.with(|slot| {
            *slot.borrow_mut() = self.0.take();
        });
    }
}

impl FrameStaging {
    pub fn new(queue: wgpu::Queue) -> Self {
        Self { inner: Arc::new(Mutex::new(RingState { queue, buffer: None, cursor: 0, peak: 0 })) }
    }

    /// Resets the write cursor at the start of a top-level frame, growing the ring
    /// once if the previous frame overflowed its backing buffer.
    pub fn begin_frame(&self, device: &wgpu::Device) {
        let mut state = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let needed = state.peak.max(INITIAL_RING_BYTES).next_power_of_two();
        if state.buffer.as_ref().is_none_or(|b| b.size() < needed) {
            let max_buf = device.limits().max_buffer_size;
            let size = needed.min(max_buf);
            if size >= COPY_ALIGN {
                state.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("frame_staging_ring"),
                    size,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
        }
        state.cursor = 0;
        state.peak = 0;
    }

    /// Runs `f` with this staging ring active on the current thread.
    pub fn with_active<R>(&self, f: impl FnOnce() -> R) -> R {
        let prev = ACTIVE_STAGING.with(|slot| slot.borrow_mut().replace(self.clone()));
        let _guard = ActiveGuard(prev);
        f()
    }

    fn copy_into(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::Buffer,
        dst_offset: u64,
        bytes: &[u8],
    ) -> bool {
        let len = bytes.len() as u64;
        if len == 0 || !len.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
            return false;
        }
        let aligned = len.next_multiple_of(COPY_ALIGN);
        let mut state = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.buffer.is_none() {
            let size = INITIAL_RING_BYTES.max(aligned).next_power_of_two().min(device.limits().max_buffer_size);
            if size >= aligned {
                state.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("frame_staging_ring"),
                    size,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
        }
        let end = state.cursor.saturating_add(aligned);
        state.peak = state.peak.max(end);
        let Some(buf) = &state.buffer else {
            return false;
        };
        if end <= buf.size() {
            let offset = state.cursor;
            state.queue.write_buffer(buf, offset, bytes);
            encoder.copy_buffer_to_buffer(buf, offset, dst, dst_offset, len);
            state.cursor = end;
            return true;
        }
        false
    }
}

/// Encodes an encoder-ordered copy of `bytes` into `dst` at `dst_offset`, using the active
/// frame staging ring when inside a renderer submission and falling back to a transient
/// mapped staging buffer in standalone unit tests or on first-frame overflow.
pub(crate) fn encode_copy(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    dst: &wgpu::Buffer,
    dst_offset: u64,
    bytes: &[u8],
) {
    if bytes.is_empty() {
        return;
    }
    let staged = ACTIVE_STAGING.with(|slot| {
        slot.borrow().as_ref().is_some_and(|ring| ring.copy_into(device, encoder, dst, dst_offset, bytes))
    });
    if staged {
        return;
    }
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fallback_stage"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    stage.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytes);
    stage.unmap();
    encoder.copy_buffer_to_buffer(&stage, 0, dst, dst_offset, bytes.len() as u64);
}
