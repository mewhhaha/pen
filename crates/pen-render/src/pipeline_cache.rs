// Compiled pipelines kept on disk between runs, where the device can keep
// them (Vulkan, with `Features::PIPELINE_CACHE`): a pipeline made with the
// cache skips the driver's compile when an earlier run made the same one.
// One file per adapter, as `wgpu::util::pipeline_cache_key` names it; wgpu
// and the driver ignore data that does not fit, such as another driver
// version's.
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cheap to clone; every clone is the same cache.
#[derive(Clone, Default)]
pub(crate) struct PipelineCache(Option<Arc<Stored>>);

struct Stored {
    cache: wgpu::PipelineCache,
    #[cfg(not(target_arch = "wasm32"))]
    path: PathBuf,
    /// Pipelines were made with it since it was last written.
    dirty: AtomicBool,
}

impl PipelineCache {
    /// The cache kept under `directory` for the device's adapter, read from
    /// its file if there is one; none if the device cannot keep one.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(device: &wgpu::Device, directory: Option<&Path>) -> Self {
        let Some(directory) = directory else { return Self::default() };
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return Self::default();
        }
        let Some(key) = wgpu::util::pipeline_cache_key(&device.adapter_info()) else { return Self::default() };
        let path = directory.join(key);
        let data = std::fs::read(&path).ok();
        // SAFETY: the data is what `get_data` returned in an earlier run,
        // written whole (renamed into place); `fallback` makes wgpu start
        // empty when it is not for this adapter, driver or wgpu.
        let cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("pipelines"),
                data: data.as_deref(),
                fallback: true,
            })
        };
        Self(Some(Arc::new(Stored { cache, path, dirty: AtomicBool::new(false) })))
    }

    /// Browsers keep their own.
    #[cfg(target_arch = "wasm32")]
    pub fn open(_device: &wgpu::Device, _directory: Option<&Path>) -> Self {
        Self::default()
    }

    /// The cache, for a pipeline about to be made with it.
    pub fn get(&self) -> Option<&wgpu::PipelineCache> {
        let stored = self.0.as_ref()?;
        stored.dirty.store(true, Ordering::Relaxed);
        Some(&stored.cache)
    }

    /// Writes it to its file if pipelines were made since it was last
    /// written: slow enough (a few megabytes) to belong off the frame thread.
    pub fn save(&self) {
        if let Some(stored) = &self.0
            && stored.dirty.swap(false, Ordering::Relaxed)
        {
            stored.write();
        }
    }
}

impl Stored {
    #[cfg(not(target_arch = "wasm32"))]
    fn write(&self) {
        let Some(data) = self.cache.get_data() else { return };
        // Written beside it, then renamed over it, so that a run starting
        // meanwhile reads the old file or the new one, never part of one.
        let partial = self.path.with_extension(format!("{}.partial", std::process::id()));
        let written = (|| {
            if let Some(directory) = self.path.parent() {
                std::fs::create_dir_all(directory)?;
            }
            std::fs::write(&partial, &data)?;
            std::fs::rename(&partial, &self.path)
        })();
        if let Err(error) = written {
            let _ = std::fs::remove_file(&partial);
            log::warn!("pipeline cache {}: {error}", self.path.display());
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn write(&self) {}
}

impl Drop for Stored {
    fn drop(&mut self) {
        if *self.dirty.get_mut() {
            self.write();
        }
    }
}
