// Results the GPU works out that the renderer reads back (pass timestamps,
// how many instances culling kept): copied into one of a few buffers at the
// end of a frame and mapped once the GPU is done with it, a frame or more
// later, so that reading them never waits on the GPU.
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

// A buffer's state.
const FREE: u8 = 0;
const MAPPING: u8 = 1;
const MAPPED: u8 = 2;
const FAILED: u8 = 3;

struct Slot<T> {
    buffer: wgpu::Buffer,
    /// How many bytes of it the frame copied in.
    bytes: u64,
    /// What the frame needs to make sense of them.
    about: T,
    state: Arc<AtomicU8>,
}

/// A few frames' readback buffers, each with `T` saying what its bytes are.
pub(crate) struct Readbacks<T> {
    label: &'static str,
    slots: Vec<Slot<T>>,
}

impl<T: Default> Readbacks<T> {
    pub fn new(label: &'static str, count: usize, device: &wgpu::Device) -> Self {
        let slots = (0..count)
            .map(|_| Slot {
                buffer: buffer(device, label, 256),
                bytes: 0,
                about: T::default(),
                state: Arc::new(AtomicU8::new(FREE)),
            })
            .collect();
        Self { label, slots }
    }

    /// Hands each buffer that has come back to `read`, with what it is
    /// about, and frees it.
    pub fn take(&mut self, mut read: impl FnMut(&[u8], &T)) {
        for slot in &mut self.slots {
            match slot.state.load(Ordering::Acquire) {
                MAPPED => {
                    if let Ok(mapped) = slot.buffer.slice(..slot.bytes).get_mapped_range() {
                        read(&mapped, &slot.about);
                    }
                    slot.buffer.unmap();
                    slot.state.store(FREE, Ordering::Release);
                }
                FAILED => slot.state.store(FREE, Ordering::Release),
                _ => {}
            }
        }
    }

    /// A free buffer of at least `bytes` (made anew if it is smaller), if
    /// one is free: the frame copies into it, then calls `submitted`.
    pub fn free(&mut self, device: &wgpu::Device, bytes: u64) -> Option<usize> {
        let index = self.slots.iter().position(|slot| slot.state.load(Ordering::Acquire) == FREE)?;
        let slot = &mut self.slots[index];
        if slot.buffer.size() < bytes {
            slot.buffer = buffer(device, self.label, bytes.next_power_of_two().max(256));
        }
        Some(index)
    }

    pub fn buffer(&self, index: usize) -> &wgpu::Buffer {
        &self.slots[index].buffer
    }

    /// Maps buffer `index`, whose first `bytes` the frame just submitted
    /// copies into, to be read once the GPU is done with it.
    pub fn submitted(&mut self, index: usize, bytes: u64, about: T) {
        let slot = &mut self.slots[index];
        slot.bytes = bytes;
        slot.about = about;
        slot.state.store(MAPPING, Ordering::Release);
        let state = slot.state.clone();
        slot.buffer.slice(..bytes).map_async(wgpu::MapMode::Read, move |result| {
            state.store(if result.is_ok() { MAPPED } else { FAILED }, Ordering::Release);
        });
    }
}

fn buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}
