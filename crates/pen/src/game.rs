//! Native scheduling adapter for the shared game runtime: the runtime's
//! methods that wait run to completion here, the program answering on its
//! own thread.
use crate::program::NativeBackend;
use crate::snapshot::{self, SavedState};
use anyhow::Result;
use blot_host::GuestModule;
use pen_protocol::{Catalog, RenderFrame};
pub use pen_runtime::game::{Edits, FrameTimings, Host, InvalidFrame, Moment, Quiet, ReloadResult, hash_lanes};
pub use pen_runtime::program::Unresponsive;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use wasmtime::Engine;

type Runtime = pen_runtime::game::GameRuntime<NativeBackend>;

pub struct GameRuntime(Runtime);

/// The runtime's methods that wait, waited for.
macro_rules! blocking {
    ($($name:ident($($argument:ident: $type:ty),*) -> $output:ty;)*) => {$(
        pub fn $name(&mut self, $($argument: $type),*) -> $output {
            pollster::block_on(self.0.$name($($argument),*))
        }
    )*};
}

impl GameRuntime {
    pub fn new(
        engine: &Engine,
        module: GuestModule,
        catalog: Catalog,
        host: &mut dyn Host,
        initial_save: Option<SavedState>,
        initial_scene: Option<SavedState>,
    ) -> Result<Self> {
        pollster::block_on(Runtime::new(
            &NativeBackend(engine.clone()),
            module,
            catalog,
            host,
            initial_save,
            initial_scene,
        ))
        .map(Self)
    }

    /// Hash every packet received. Tests also keep each frame, to say how two differ.
    pub fn hash_packets(&mut self, enabled: bool) {
        self.0.hash_packets(enabled);
        #[cfg(test)]
        self.0.keep_received(enabled);
    }

    blocking! {
        receive(host: &dyn Host) -> Result<RenderFrame>;
        follow(host: &mut dyn Host, frame: &RenderFrame) -> Result<()>;
        reload(host: &mut dyn Host, module: GuestModule, catalog: Catalog) -> ReloadResult;
        snapshot() -> Result<SavedState>;
        moment() -> Result<Moment>;
        restore(host: &mut dyn Host, moment: &Moment) -> Result<RenderFrame>;
        load(host: &mut dyn Host, saved: &SavedState) -> Result<()>;
        load_scene(host: &mut dyn Host, saved: &SavedState) -> Result<()>;
    }

    pub fn close(&mut self) {
        pollster::block_on(self.0.close());
    }

    pub fn save(&mut self, path: &Path) -> Result<()> {
        snapshot::write(path, &self.snapshot()?)
    }
    pub fn save_scene(&mut self, path: &Path) -> Result<()> {
        snapshot::write(path, &pollster::block_on(self.0.scene())?)
    }
}

/// The runtime's methods that do not wait.
impl Deref for GameRuntime {
    type Target = Runtime;
    fn deref(&self) -> &Runtime {
        &self.0
    }
}

impl DerefMut for GameRuntime {
    fn deref_mut(&mut self) -> &mut Runtime {
        &mut self.0
    }
}
