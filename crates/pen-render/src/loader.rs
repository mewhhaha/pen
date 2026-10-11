// The asset store's background: work that reads asset files under the asset
// root and decodes them (JSON, packed texture levels, PNG and JPEG), and
// compiles shaders and their pipelines, as the TS does with promises and
// `createRenderPipelineAsync`. What it brings goes back over a channel to the
// thread that submits frames, which uploads it to the GPU. Natively it runs
// on threads; in the browser, as tasks on the page's event loop.
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use anyhow::Result;

use crate::asset_store::{AssetKind, Material, Mesh, parse_material, parse_mesh, validate_asset_key};
use crate::packed_texture::{PackedHeader, PackedLevel, PackedReader, is_packed_texture};
use crate::platform;
use crate::shader::{self, PipelineContext, Pipelines};
use crate::texture::DecodedImage;

/// Insets draw with this many samples a pixel.
pub(crate) const INSET_SAMPLES: u32 = 4;

pub(crate) enum Job {
    Load { slot: usize, kind: AssetKind, key: String, revision: u64, early: bool },
    /// Read a shader to see whether it draws sprites or meshes.
    Sort { key: String },
    /// Compile a shader's pipelines for another sample count.
    Compile { slot: usize, revision: u64, samples: u32, module: wgpu::ShaderModule, instancing: bool },
}

#[derive(Clone, Copy)]
pub(crate) enum Priority {
    /// Asked for by a frame.
    Urgent,
    /// Ahead of need.
    Background,
}

pub(crate) enum Message {
    Loaded { slot: usize, revision: u64, early: bool, result: Result<Loaded, String> },
    Sorted { key: String, sprite: bool },
    Compiled { slot: usize, revision: u64, samples: u32, result: Result<Pipelines, String> },
    /// A shader's opaque pipeline for insets, compiled after it loaded.
    Inset { slot: usize, revision: u64, result: Result<wgpu::RenderPipeline, String> },
}

pub(crate) enum Loaded {
    Mesh(Mesh),
    Material(Material),
    Shader { module: wgpu::ShaderModule, instancing: bool, samples: u32, pipelines: Pipelines },
    SpriteShader(wgpu::RenderPipeline),
    Image(DecodedImage),
    /// A packed texture: its header now, its levels as each is inflated.
    Packed { header: PackedHeader, levels: Receiver<LevelMessage> },
}

pub(crate) enum LevelMessage {
    Level(PackedLevel),
    Done,
    Failed(String),
}

#[cfg(not(target_arch = "wasm32"))]
mod queue {
    use std::collections::VecDeque;
    use std::sync::{Arc, Condvar, Mutex};

    use super::{Job, Priority};

    struct Queues {
        urgent: VecDeque<Job>,
        background: VecDeque<Job>,
        closed: bool,
    }

    /// Jobs for the loader's threads, what a frame asks for first.
    #[derive(Clone)]
    pub(crate) struct JobQueue {
        inner: Arc<(Mutex<Queues>, Condvar)>,
    }

    impl JobQueue {
        pub(super) fn new() -> Self {
            Self {
                inner: Arc::new((
                    Mutex::new(Queues { urgent: VecDeque::new(), background: VecDeque::new(), closed: false }),
                    Condvar::new(),
                )),
            }
        }

        pub fn push(&self, job: Job, priority: Priority) {
            let (queues, ready) = &*self.inner;
            let mut queues = queues.lock().unwrap();
            match priority {
                Priority::Urgent => queues.urgent.push_back(job),
                Priority::Background => queues.background.push_back(job),
            }
            ready.notify_one();
        }

        pub(super) fn pop(&self) -> Option<Job> {
            let (queues, ready) = &*self.inner;
            let mut queues = queues.lock().unwrap();
            loop {
                if queues.closed {
                    return None;
                }
                if let Some(job) = queues.urgent.pop_front().or_else(|| queues.background.pop_front()) {
                    return Some(job);
                }
                queues = ready.wait(queues).unwrap();
            }
        }

        pub(super) fn close(&self) {
            let (queues, ready) = &*self.inner;
            let mut queues = queues.lock().unwrap();
            queues.closed = true;
            queues.urgent.clear();
            queues.background.clear();
            ready.notify_all();
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod queue {
    use super::{Job, Priority, Worker};

    /// Jobs for the page's event loop: each starts at once, as the TS's
    /// promises do (the store already keeps prefetches to a few at a time).
    #[derive(Clone)]
    pub(crate) struct JobQueue {
        pub(super) worker: Worker,
    }

    impl JobQueue {
        pub fn push(&self, job: Job, _priority: Priority) {
            let worker = self.worker.clone();
            wasm_bindgen_futures::spawn_local(async move { worker.run(job).await });
        }
    }
}

pub(crate) use queue::JobQueue;

/// What the loader's work shares: where assets are, how to compile, and where
/// to send what it brings.
#[derive(Clone)]
pub(crate) struct Worker {
    /// The asset root: a directory, or in the browser a URL.
    root: Arc<str>,
    context: PipelineContext,
    /// The sample count frames draw with now, which a shader's first
    /// pipelines are compiled for.
    samples_now: Arc<AtomicU32>,
    closed: Arc<AtomicBool>,
    results: Sender<Message>,
}

pub(crate) struct Loader {
    pub jobs: JobQueue,
    pub results: Receiver<Message>,
    pub samples_now: Arc<AtomicU32>,
    closed: Arc<AtomicBool>,
    #[cfg(not(target_arch = "wasm32"))]
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Loader {
    pub fn new(root: PathBuf, context: PipelineContext) -> Result<Self> {
        let (sender, results) = channel();
        let samples_now = Arc::new(AtomicU32::new(1));
        let closed = Arc::new(AtomicBool::new(false));
        let worker = Worker {
            root: root.to_string_lossy().into(),
            context,
            samples_now: samples_now.clone(),
            closed: closed.clone(),
            results: sender,
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let jobs = JobQueue::new();
            let count = std::thread::available_parallelism().map_or(4, |count| count.get()).clamp(2, 6);
            let mut threads = Vec::with_capacity(count);
            for index in 0..count {
                let (worker, jobs) = (worker.clone(), jobs.clone());
                threads.push(std::thread::Builder::new().name(format!("pen-assets-{index}")).spawn(move || {
                    while let Some(job) = jobs.pop() {
                        pollster::block_on(worker.run(job));
                    }
                })?);
            }
            Ok(Self { jobs, results, samples_now, closed, threads })
        }
        #[cfg(target_arch = "wasm32")]
        Ok(Self { jobs: JobQueue { worker }, results, samples_now, closed })
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Relaxed);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.jobs.close();
            // Wait out any job already inside the GPU driver: exiting the
            // process while a worker compiles a pipeline crashes in the
            // driver's teardown.
            for thread in self.threads.drain(..) {
                let _ = thread.join();
            }
        }
    }
}

impl Worker {
    fn send(&self, message: Message) {
        // The renderer is gone when this fails; nothing is left to tell.
        let _ = self.results.send(message);
    }

    async fn read(&self, key: &str) -> Result<Vec<u8>> {
        platform::read_asset(&self.root, key).await
    }

    async fn run(&self, job: Job) {
        if self.closed.load(Ordering::Relaxed) {
            return;
        }
        match job {
            Job::Load { slot, kind, key, revision, early } => self.load(slot, kind, &key, revision, early).await,
            Job::Sort { key } => {
                let Ok(bytes) = self.read(&key).await else { return };
                let Ok(code) = std::str::from_utf8(&bytes) else { return };
                if let Some(sprite) = shader::draws_sprites(code) {
                    self.send(Message::Sorted { key, sprite });
                }
            }
            Job::Compile { slot, revision, samples, module, instancing } => {
                let result = shader::checked(&self.context.device, || {
                    shader::pipelines(&self.context, &module, instancing, samples)
                })
                .await;
                self.send(Message::Compiled { slot, revision, samples, result });
            }
        }
    }

    async fn load(&self, slot: usize, kind: AssetKind, key: &str, revision: u64, early: bool) {
        let loaded = |result: Result<Loaded>| Message::Loaded {
            slot,
            revision,
            early,
            result: result.map_err(|error| format!("{error:#}")),
        };
        let bytes = match self.read(key).await {
            Ok(bytes) => bytes,
            Err(error) => return self.send(loaded(Err(error))),
        };
        let device = &self.context.device;
        match kind {
            AssetKind::Mesh => self.send(loaded(parse_mesh(&bytes).map(Loaded::Mesh))),
            AssetKind::Material => {
                let material = parse_material(&bytes).and_then(|material| {
                    validate_asset_key(&material.shader)?;
                    for key in [&material.texture, &material.normal_texture, &material.layer_texture].into_iter().flatten() {
                        validate_asset_key(key)?;
                    }
                    Ok(material)
                });
                self.send(loaded(material.map(Loaded::Material)));
            }
            AssetKind::SpriteShader => {
                let pipeline = match shader::compile_shader(device, &bytes).await {
                    Ok((module, _)) => shader::checked(device, || shader::sprite_pipeline(&self.context, &module))
                        .await
                        .map_err(anyhow::Error::msg),
                    Err(error) => Err(error),
                };
                self.send(loaded(pipeline.map(Loaded::SpriteShader)));
            }
            AssetKind::Shader => {
                let (module, code) = match shader::compile_shader(device, &bytes).await {
                    Ok(compiled) => compiled,
                    Err(error) => return self.send(loaded(Err(error))),
                };
                let instancing = shader::has_instancing(&code);
                // The current sample count's pipelines compile now; another
                // count's when the settings first ask for it.
                let samples = self.samples_now.load(Ordering::Relaxed);
                let pipelines =
                    shader::checked(device, || shader::pipelines(&self.context, &module, instancing, samples)).await;
                let pipelines = match pipelines {
                    Ok(pipelines) => pipelines,
                    Err(message) => return self.send(loaded(Err(anyhow::Error::msg(message)))),
                };
                self.send(loaded(Ok(Loaded::Shader { module: module.clone(), instancing, samples, pipelines })));
                // Insets draw with 4 samples a pixel: their pipeline compiles
                // in the background now, so an inset first shown never waits
                // on it.
                if samples != INSET_SAMPLES {
                    platform::yield_now().await;
                    let result = shader::checked(device, || {
                        shader::variant(&self.context, &module, INSET_SAMPLES, true, wgpu::CompareFunction::Less, None, false)
                    })
                    .await;
                    self.send(Message::Inset { slot, revision, result });
                }
            }
            AssetKind::Texture => {
                if !is_packed_texture(&bytes) {
                    // An image, whole, its smaller levels drawn from it.
                    return self.send(loaded(platform::decode_image(bytes).await.map(Loaded::Image)));
                }
                let mut reader = match PackedReader::open(bytes) {
                    Ok(reader) => reader,
                    Err(error) => return self.send(loaded(Err(error))),
                };
                let (levels, receiver) = channel();
                self.send(loaded(Ok(Loaded::Packed { header: reader.header, levels: receiver })));
                // Each level, smallest first, as soon as it is inflated, until
                // the texture is done with or the renderer closes.
                loop {
                    if self.closed.load(Ordering::Relaxed) {
                        return;
                    }
                    let message = match reader.next_level() {
                        Ok(Some(level)) => LevelMessage::Level(level),
                        Ok(None) => LevelMessage::Done,
                        Err(error) => LevelMessage::Failed(format!("{error:#}")),
                    };
                    let last = !matches!(message, LevelMessage::Level(_));
                    if levels.send(message).is_err() || last {
                        return;
                    }
                    platform::yield_now().await;
                }
            }
        }
    }
}
