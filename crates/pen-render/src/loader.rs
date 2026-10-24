// The asset store's background: work that reads asset files under the asset
// root and decodes them (JSON, packed texture levels, PNG and JPEG), and
// compiles shaders and pipelines. What it brings goes back over a channel to
// the thread that submits frames, which uploads it to the GPU. Natively it
// runs on threads, which renderers sharing a device share too; in the
// browser, as tasks on the page's event loop.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::asset_store::{AssetKind, Material, Mesh, parse_material, parse_mesh, validate_asset_key};
use crate::built_in::BuiltIn;
use crate::compressed::{self, LevelReader};
use crate::packed_texture::{PackedHeader, PackedLevel};
use crate::platform;
use crate::prefetch::{Prefetch, Prepared};
use crate::shader::{self, PipelineContext, Pipelines, Stage, Variants};
use crate::texture::DecodedImage;

pub(crate) enum Job {
    Load {
        slot: usize,
        kind: AssetKind,
        key: String,
        revision: u64,
        early: bool,
    },
    /// Read a shader to see whether it draws sprites or meshes.
    Sort {
        key: String,
    },
    /// Compile some of a shader's pipelines for a sample count.
    Compile {
        slot: usize,
        revision: u64,
        samples: u32,
        variants: Variants,
        module: wgpu::ShaderModule,
        instancing: bool,
        skinning: bool,
        persistent: bool,
    },
    /// Compile a depth-only pipeline, plain, instanced or skinned, once a
    /// loaded material can cast shadows.
    Shadow {
        slot: usize,
        revision: u64,
        module: wgpu::ShaderModule,
        stage: Stage,
    },
    BuiltIn(BuiltIn),
    /// Compile the pipeline that culls instances on the GPU.
    Culling,
    /// Write the pipeline cache to its file.
    Persist,
}

#[derive(Clone, Copy)]
pub(crate) enum Priority {
    /// Asked for by a frame.
    Urgent,
    /// Ahead of need.
    Background,
}

pub(crate) enum Message {
    Loaded {
        slot: usize,
        revision: u64,
        early: bool,
        result: Result<Loaded, String>,
    },
    Sorted {
        key: String,
        sprite: bool,
    },
    Compiled {
        slot: usize,
        revision: u64,
        samples: u32,
        variants: Variants,
        result: Result<Pipelines, String>,
    },
    Shadow {
        slot: usize,
        revision: u64,
        stage: Stage,
        result: Result<wgpu::RenderPipeline, String>,
    },
    BuiltIn {
        which: BuiltIn,
        result: Result<wgpu::RenderPipeline, String>,
    },
    Culling(Result<wgpu::ComputePipeline, String>),
    /// A packed texture's level went down its stream.
    Level,
}

pub(crate) enum Loaded {
    Mesh(Mesh),
    Material(Material),
    /// A shader's module; its pipelines compile in jobs of their own.
    Shader {
        module: wgpu::ShaderModule,
        instancing: bool,
        skinning: bool,
        persistent: bool,
        /// Its `bounds_margin`, if it declares one.
        margin: Option<f32>,
        /// Whether it reads the cascaded shadow maps.
        cascaded: bool,
    },
    SpriteShader(wgpu::RenderPipeline),
    Image(DecodedImage),
    /// A packed texture: its header now, its levels as each is inflated.
    Packed {
        header: PackedHeader,
        levels: Receiver<LevelMessage>,
    },
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

    use super::{Job, Priority, Worker};

    struct Queues {
        urgent: VecDeque<(Worker, Job)>,
        background: VecDeque<(Worker, Job)>,
        closed: bool,
    }

    /// The jobs waiting for the loader's threads, what a frame asks for first.
    struct Waiting {
        queues: Mutex<Queues>,
        ready: Condvar,
    }

    impl Waiting {
        fn pop(&self) -> Option<(Worker, Job)> {
            let mut queues = self.queues.lock().unwrap();
            loop {
                if queues.closed {
                    return None;
                }
                if let Some(job) = queues.urgent.pop_front().or_else(|| queues.background.pop_front()) {
                    return Some(job);
                }
                queues = self.ready.wait(queues).unwrap();
            }
        }
    }

    /// The threads, which end with the last renderer using them.
    struct Threads {
        waiting: Arc<Waiting>,
        handles: Vec<std::thread::JoinHandle<()>>,
    }

    impl Drop for Threads {
        fn drop(&mut self) {
            let mut queues = self.waiting.queues.lock().unwrap();
            queues.closed = true;
            queues.urgent.clear();
            queues.background.clear();
            drop(queues);
            self.waiting.ready.notify_all();
            // Wait out any job already inside the GPU driver: exiting the
            // process while a worker compiles a pipeline crashes in the
            // driver's teardown.
            for thread in self.handles.drain(..) {
                let _ = thread.join();
            }
        }
    }

    /// One renderer's jobs for the threads.
    #[derive(Clone)]
    pub(crate) struct JobQueue {
        worker: Worker,
        threads: Arc<Threads>,
    }

    impl JobQueue {
        pub(super) fn new(worker: Worker) -> std::io::Result<Self> {
            let waiting = Arc::new(Waiting {
                queues: Mutex::new(Queues { urgent: VecDeque::new(), background: VecDeque::new(), closed: false }),
                ready: Condvar::new(),
            });
            let count = std::thread::available_parallelism().map_or(4, |count| count.get()).clamp(2, 6);
            let mut threads = Threads { waiting: waiting.clone(), handles: Vec::with_capacity(count) };
            for index in 0..count {
                let waiting = waiting.clone();
                threads.handles.push(std::thread::Builder::new().name(format!("pen-assets-{index}")).spawn(
                    move || {
                        while let Some((worker, job)) = waiting.pop() {
                            pollster::block_on(worker.run(job));
                        }
                    },
                )?);
            }
            Ok(Self { worker, threads: Arc::new(threads) })
        }

        /// A queue for another renderer's work, on the same threads.
        pub(super) fn sharing(&self, worker: Worker) -> Self {
            Self { worker, threads: self.threads.clone() }
        }

        pub fn push(&self, job: Job, priority: Priority) {
            let waiting = &self.threads.waiting;
            let mut queues = waiting.queues.lock().unwrap();
            let item = (self.worker.clone(), job);
            match priority {
                Priority::Urgent => queues.urgent.push_back(item),
                Priority::Background => queues.background.push_back(item),
            }
            drop(queues);
            waiting.ready.notify_one();
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod queue {
    use super::{Job, Priority, Worker};

    /// Jobs for the page's event loop: each starts at once, as promises do
    /// (the store already keeps prefetches to a few at a time).
    #[derive(Clone)]
    pub(crate) struct JobQueue {
        worker: Worker,
    }

    impl JobQueue {
        pub(super) fn new(worker: Worker) -> std::io::Result<Self> {
            Ok(Self { worker })
        }

        pub(super) fn sharing(&self, worker: Worker) -> Self {
            Self { worker }
        }

        pub fn push(&self, job: Job, _priority: Priority) {
            let worker = self.worker.clone();
            wasm_bindgen_futures::spawn_local(async move { worker.run(job).await });
        }
    }
}

pub(crate) use queue::JobQueue;

/// What one renderer's background work shares: where its assets are, how to
/// compile, and where to send what it brings.
#[derive(Clone)]
pub(crate) struct Worker {
    /// The asset root: a directory, or in the browser a URL.
    root: Arc<str>,
    context: PipelineContext,
    closed: Arc<AtomicBool>,
    /// What loaded materials need of their shaders, updated as each loads,
    /// before its shader loads.
    needs: Arc<Mutex<Needs>>,
    results: Sender<Message>,
    /// Assets read and decoded before the renderer was made, for their
    /// first loads.
    prefetch: Arc<Mutex<Option<Prefetch>>>,
}

/// What a material needs of its shader: which variants, and whether it
/// casts shadows.
#[derive(Clone, Copy, Default)]
pub(crate) struct Need {
    pub variants: Variants,
    pub shadow: bool,
}

/// Each material's need, by its shader.
#[derive(Default)]
pub(crate) struct Needs {
    /// Each material key's shader key.
    shaders: HashMap<String, String>,
    by_shader: HashMap<String, HashMap<String, Need>>,
}

impl Needs {
    fn set(&mut self, material: &str, shader: &str, need: Need) {
        if let Some(previous) = self.shaders.insert(material.to_owned(), shader.to_owned())
            && previous != shader
            && let Some(materials) = self.by_shader.get_mut(&previous)
        {
            materials.remove(material);
        }
        self.by_shader.entry(shader.to_owned()).or_default().insert(material.to_owned(), need);
    }

    /// What the materials naming `shader` need of it.
    pub fn of(&self, shader: &str) -> Need {
        self.by_shader.get(shader).into_iter().flat_map(HashMap::values).fold(Need::default(), |all, need| Need {
            variants: all.variants | need.variants,
            shadow: all.shadow || need.shadow,
        })
    }
}

pub(crate) struct Loader {
    pub jobs: JobQueue,
    pub results: Receiver<Message>,
    pub needs: Arc<Mutex<Needs>>,
    pub prefetch: Arc<Mutex<Option<Prefetch>>>,
    closed: Arc<AtomicBool>,
}

impl Loader {
    pub fn new(root: PathBuf, context: PipelineContext) -> Result<Self> {
        let (worker, loader) = Self::parts(root, context);
        Ok(loader(JobQueue::new(worker)?))
    }

    /// A loader of assets under `root` for another renderer, on this one's
    /// threads.
    pub fn sharing(&self, root: PathBuf, context: PipelineContext) -> Self {
        let (worker, loader) = Self::parts(root, context);
        loader(self.jobs.sharing(worker))
    }

    fn parts(root: PathBuf, context: PipelineContext) -> (Worker, impl FnOnce(JobQueue) -> Loader) {
        let (sender, results) = channel();
        let closed = Arc::new(AtomicBool::new(false));
        let needs = Arc::new(Mutex::new(Needs::default()));
        let prefetch = Arc::new(Mutex::new(None));
        let worker = Worker {
            root: root.to_string_lossy().into(),
            context,
            closed: closed.clone(),
            needs: needs.clone(),
            results: sender,
            prefetch: prefetch.clone(),
        };
        (worker, move |jobs| Loader { jobs, results, needs, prefetch, closed })
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        // Its jobs still queued, and textures still streaming, stop; the
        // threads end with the last renderer using them.
        self.closed.store(true, Ordering::Relaxed);
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

    /// The first of `keys` there is, or why the last is not.
    async fn read_first(&self, keys: &[String]) -> Result<Vec<u8>> {
        let (last, rest) = keys.split_last().expect("a key to read");
        for key in rest {
            if let Ok(bytes) = self.read(key).await {
                return Ok(bytes);
            }
        }
        self.read(last).await
    }

    async fn run(&self, job: Job) {
        if self.closed.load(Ordering::Relaxed) {
            return;
        }
        let device = &self.context.device;
        match job {
            Job::Load { slot, kind, key, revision, early } => self.load(slot, kind, &key, revision, early).await,
            Job::Sort { key } => {
                let Ok(bytes) = self.read(&key).await else { return };
                let Ok(code) = std::str::from_utf8(&bytes) else { return };
                if let Some(sprite) = shader::draws_sprites(code) {
                    self.send(Message::Sorted { key, sprite });
                }
            }
            Job::Compile { slot, revision, samples, variants, module, instancing, skinning, persistent } => {
                let result = self.compile(&module, instancing, skinning, persistent, samples, variants).await;
                self.send(Message::Compiled { slot, revision, samples, variants, result });
            }
            Job::Shadow { slot, revision, module, stage } => {
                let clock = platform::Clock::new();
                let result = shader::checked(device, || shader::shadow_pipeline(&self.context, &module, stage)).await;
                log::debug!("shadow pipeline: slot {slot}, {stage:?}, {:.2} ms", clock.now() * 1000.0);
                self.send(Message::Shadow { slot, revision, stage, result });
            }
            Job::BuiltIn(which) => {
                let result = shader::checked(device, || which.compile(&self.context)).await;
                self.send(Message::BuiltIn { which, result });
            }
            Job::Culling => {
                let result = shader::checked(device, || crate::renderer::culling_pipeline(&self.context)).await;
                self.send(Message::Culling(result));
            }
            Job::Persist => self.context.cache.save(),
        }
    }

    async fn compile(
        &self,
        module: &wgpu::ShaderModule,
        instancing: bool,
        skinning: bool,
        persistent: bool,
        samples: u32,
        variants: Variants,
    ) -> Result<Pipelines, String> {
        let clock = platform::Clock::new();
        let result = shader::checked(&self.context.device, || {
            shader::pipelines(&self.context, module, instancing, skinning, persistent, samples, variants)
        })
        .await;
        log::debug!("pipelines {variants:?} at {samples} samples: {:.2} ms", clock.now() * 1000.0);
        result
    }

    /// A material whose keys stay in the asset root, its need of its shader
    /// noted.
    fn material(&self, key: &str, material: Result<Material>) -> Result<Material> {
        let material = material.and_then(|material| {
            validate_asset_key(&material.shader)?;
            for key in [&material.texture, &material.normal_texture, &material.layer_texture].into_iter().flatten() {
                validate_asset_key(key)?;
            }
            Ok(material)
        });
        if let Ok(material) = &material {
            let need = Need { variants: material.variants(), shadow: material.casts_shadow() };
            self.needs.lock().unwrap().set(key, &material.shader, need);
        }
        material
    }

    async fn load(&self, slot: usize, kind: AssetKind, key: &str, revision: u64, early: bool) {
        let loaded = |result: Result<Loaded>| Message::Loaded {
            slot,
            revision,
            early,
            result: result.map_err(|error| format!("{error:#}")),
        };
        // An asset's first load takes what was read and decoded ahead.
        let prepared = if revision == 1 {
            self.prefetch.lock().unwrap().as_ref().and_then(|prefetch| prefetch.take(kind, key))
        } else {
            None
        };
        let bytes = match prepared {
            Some(Prepared::Bytes(bytes)) => bytes,
            Some(Prepared::Mesh(mesh)) => return self.send(loaded(Ok(Loaded::Mesh(*mesh)))),
            Some(Prepared::Material(material)) => {
                return self.send(loaded(self.material(key, Ok(material)).map(Loaded::Material)));
            }
            Some(Prepared::Image(image)) => return self.send(loaded(Ok(Loaded::Image(image)))),
            Some(Prepared::Packed(header, levels)) => {
                let (sender, receiver) = channel();
                self.send(loaded(Ok(Loaded::Packed { header, levels: receiver })));
                for level in levels.into_iter().map(LevelMessage::Level).chain([LevelMessage::Done]) {
                    if sender.send(level).is_err() {
                        return;
                    }
                    self.send(Message::Level);
                }
                return;
            }
            None => {
                // A KTX2 texture's variant the device samples, if there is one.
                let bytes = if kind == AssetKind::Texture {
                    self.read_first(&compressed::candidates(key, self.context.device.features())).await
                } else {
                    self.read(key).await
                };
                match bytes {
                    Ok(bytes) => bytes,
                    Err(error) => return self.send(loaded(Err(error))),
                }
            }
        };
        let device = &self.context.device;
        match kind {
            AssetKind::Mesh => self.send(loaded(parse_mesh(&bytes).map(Loaded::Mesh))),
            AssetKind::Material => {
                let material = self.material(key, parse_material(&bytes));
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
                let shader = shader::compile_shader(device, &bytes).await.map(|(module, code)| {
                    let instancing = shader::has_instancing(&code);
                    let skinning = shader::has_skinning(&code);
                    let persistent = shader::has_persistent(&code);
                    let cascaded = code.contains("shadow_cascades");
                    Loaded::Shader {
                        module,
                        instancing,
                        skinning,
                        persistent,
                        margin: shader::bounds_margin(&code),
                        cascaded,
                    }
                });
                self.send(loaded(shader));
            }
            AssetKind::Texture => {
                if !compressed::has_levels(&bytes) {
                    // An image, whole, its smaller levels drawn from it.
                    return self.send(loaded(platform::decode_image(bytes).await.map(Loaded::Image)));
                }
                let mut reader = match LevelReader::open(bytes, device.features()) {
                    Ok(reader) => reader,
                    Err(error) => return self.send(loaded(Err(error))),
                };
                let (levels, receiver) = channel();
                self.send(loaded(Ok(Loaded::Packed { header: reader.header(), levels: receiver })));
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
                    if levels.send(message).is_err() {
                        return;
                    }
                    // For whoever waits on the loader.
                    self.send(Message::Level);
                    if last {
                        return;
                    }
                    platform::yield_now().await;
                }
            }
        }
    }
}
