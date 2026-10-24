//! Controlled access for the separate session-test guest. No desktop or live
//! clipboard, URL, Steam or report service is opened by these sessions.
use super::*;
use anyhow::ensure;

impl Session {
    pub fn testing(
        options: Options,
        paths: Paths,
        engine: wasmtime::Engine,
        module: GuestModule,
        catalog: Catalog,
        gpu: Gpu,
    ) -> Result<Self> {
        let mut services = Services::new();
        for code in [pen_protocol::service::STEAM, pen_protocol::service::CLIPBOARD, pen_protocol::service::URL] {
            services.register(Box::new(pen_runtime::services::Unavailable(code)));
        }
        services.register(Box::new(pen_runtime::services::Platform {
            os: "test".into(),
            locale: Some("en-US".into()),
            device: "test".into(),
            handheld: false,
        }));
        services.register(Box::new(pen_runtime::services::Pathfinding));
        pen_simulation::register(&mut services);
        Self::with_services(options, paths, engine, module, catalog, gpu, None, services)
    }

    pub fn test_snapshot(&mut self) -> Result<SavedState> {
        self.runtime.snapshot()
    }

    /// Asset work completes while simulation time stays fixed. Loading events
    /// remain queued for the next ordinary recorded input frame.
    pub fn test_settle(&mut self) -> Result<()> {
        if self.gpu.renderer.quiet() && self.gpu.renderer.pending() == 0 {
            return Ok(());
        }
        let frame = self.last.as_ref().context("the game has not drawn a frame")?;
        settle(&mut self.gpu, frame, self.runtime.catalog())?;
        ensure!(self.gpu.renderer.quiet() && self.gpu.renderer.pending() == 0, "renderer did not settle");
        Ok(())
    }

    pub fn test_screenshot(&mut self, path: PathBuf) -> Result<()> {
        self.test_settle()?;
        self.capture.screenshots.push(path);
        self.still = false;
        self.present_last(None)
    }
}
