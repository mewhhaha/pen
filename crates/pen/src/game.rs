//! Native scheduling adapter for the shared game runtime.
use std::path::Path;
use anyhow::Result;
use blot_host::GuestModule;
use pen_protocol::{Catalog, Frame, RenderFrame};
use wasmtime::Engine;
use crate::program::NativeBackend;
use crate::snapshot::{self, SavedState};
pub use pen_runtime::game::{Edits, FrameTimings, Host, InvalidFrame, Moment, Quiet, ReloadResult, hash_lanes};

pub struct GameRuntime(pen_runtime::game::GameRuntime<NativeBackend>);
impl GameRuntime {
    pub fn new(engine: &Engine, module: GuestModule, catalog: Catalog, host: &mut dyn Host,
        initial_save: Option<SavedState>, initial_scene: Option<SavedState>) -> Result<Self> {
        pollster::block_on(pen_runtime::game::GameRuntime::new(&NativeBackend(engine.clone()), module,
            catalog, host, initial_save, initial_scene)).map(Self)
    }
    pub fn measure_frames(&mut self, enabled: bool) { self.0.measure_frames(enabled); }
    pub fn last_timings(&self) -> Option<FrameTimings> { self.0.last_timings() }
    pub fn hash_packets(&mut self, enabled: bool) { self.0.hash_packets(enabled); }
    pub fn packet_hash(&self) -> u64 { self.0.packet_hash() }
    #[cfg(test)]
    pub fn received(&self) -> Option<&RenderFrame> { self.0.received() }
    pub fn input(&mut self, frame: &Frame) -> Result<Vec<f32>> { self.0.input(frame) }
    pub fn continue_clock(&mut self, timestamp: f64) { self.0.continue_clock(timestamp); }
    pub fn send(&mut self, input: &[f32]) -> Result<()> { self.0.send(input) }
    pub fn receive(&mut self, host: &dyn Host) -> Result<RenderFrame> { pollster::block_on(self.0.receive(host)) }
    pub fn follow(&mut self, host: &mut dyn Host, frame: &RenderFrame) -> Result<()> { pollster::block_on(self.0.follow(host, frame)) }
    pub fn reload(&mut self, host: &mut dyn Host, module: GuestModule, catalog: Catalog) -> ReloadResult {
        pollster::block_on(self.0.reload(host, module, catalog))
    }
    pub fn take_drawn(&mut self) -> Option<RenderFrame> { self.0.take_drawn() }
    pub fn catalog(&self) -> &Catalog { self.0.catalog() }
    #[cfg(feature = "devtools")]
    pub fn memory_bytes(&self) -> usize { self.0.memory_bytes() }
    pub fn snapshot(&mut self) -> Result<SavedState> { pollster::block_on(self.0.snapshot()) }
    pub fn moment(&mut self) -> Result<Moment> { pollster::block_on(self.0.moment()) }
    pub fn restore(&mut self, host: &mut dyn Host, moment: &Moment) -> Result<RenderFrame> { pollster::block_on(self.0.restore(host, moment)) }
    pub fn reroot(&mut self, host: &dyn Host) -> Result<()> { pollster::block_on(self.0.reroot(host)) }
    pub fn save(&mut self, path: &Path) -> Result<()> { snapshot::write(path, &self.snapshot()?) }
    pub fn load(&mut self, host: &mut dyn Host, saved: &SavedState) -> Result<()> { pollster::block_on(self.0.load(host, saved)) }
    pub fn save_scene(&mut self, path: &Path) -> Result<()> { snapshot::write(path, &pollster::block_on(self.0.scene())?) }
    pub fn load_scene(&mut self, host: &mut dyn Host, saved: &SavedState) -> Result<()> { pollster::block_on(self.0.load_scene(host, saved)) }
    pub fn close(&mut self) { pollster::block_on(self.0.close()); }
}
