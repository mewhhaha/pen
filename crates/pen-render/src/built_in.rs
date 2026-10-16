// The renderer's own pipelines, beside its materials' shaders: the passes
// around the scene (post.rs), decals and the inset composite. They compile on
// the loader's threads ahead of need, for the sample count frames draw at,
// and on the frame's thread only when a frame needs one before it is there.
// A pipeline made first is the one kept: bind groups are made from its
// layout.
use std::collections::{HashMap, HashSet};

use crate::loader::{Job, JobQueue, Priority};
use crate::shader::PipelineContext;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum BuiltIn {
    /// Ambient occlusion from the depth of a scene of this many samples.
    Occlusion(u32),
    /// Sky and fog over a scene of this many samples.
    Atmosphere(u32),
    /// The last pass onto the output, reading the stencil of such a scene.
    Present(u32),
    BloomDown {
        first: bool,
    },
    BloomUp,
    Decal {
        samples: u32,
        paint: bool,
    },
    /// An inset's image laid over the output.
    Composite,
}

impl BuiltIn {
    /// Every one a frame drawn with `samples` per pixel may draw with.
    pub fn all(samples: u32) -> [BuiltIn; 9] {
        [
            BuiltIn::Occlusion(samples),
            BuiltIn::Atmosphere(samples),
            BuiltIn::Present(samples),
            BuiltIn::BloomDown { first: true },
            BuiltIn::BloomDown { first: false },
            BuiltIn::BloomUp,
            BuiltIn::Decal { samples, paint: true },
            BuiltIn::Decal { samples, paint: false },
            BuiltIn::Composite,
        ]
    }

    pub fn compile(self, context: &PipelineContext) -> wgpu::RenderPipeline {
        let clock = crate::platform::Clock::new();
        let pipeline = match self {
            BuiltIn::Decal { samples, paint } => crate::decals::pipeline(context, samples, paint),
            BuiltIn::Composite => crate::renderer::composite_pipeline(context),
            _ => crate::post::pipeline(context, self),
        };
        log::debug!("{self:?} pipeline: {:.2} ms", clock.now() * 1000.0);
        pipeline
    }
}

pub(crate) struct BuiltIns {
    context: PipelineContext,
    jobs: JobQueue,
    made: HashMap<BuiltIn, wgpu::RenderPipeline>,
    compiling: HashSet<BuiltIn>,
}

impl BuiltIns {
    pub fn new(context: PipelineContext, jobs: JobQueue) -> Self {
        Self { context, jobs, made: HashMap::new(), compiling: HashSet::new() }
    }

    /// Compiles these in the background, unless they are there or on their way.
    pub fn prepare(&mut self, which: impl IntoIterator<Item = BuiltIn>) {
        for which in which {
            if !self.made.contains_key(&which) && self.compiling.insert(which) {
                self.jobs.push(Job::BuiltIn(which), Priority::Urgent);
            }
        }
    }

    pub fn ready(&self, which: impl IntoIterator<Item = BuiltIn>) -> bool {
        which.into_iter().all(|which| self.made.contains_key(&which))
    }

    /// Whether some of these are not there yet but on their way.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn coming(&self, which: impl IntoIterator<Item = BuiltIn>) -> bool {
        which.into_iter().any(|which| !self.made.contains_key(&which) && self.compiling.contains(&which))
    }

    /// Background compiles under way.
    pub fn compiling(&self) -> usize {
        self.compiling.len()
    }

    /// The pipeline, compiled now if it is not there yet.
    pub fn get(&mut self, which: BuiltIn) -> &wgpu::RenderPipeline {
        let context = &self.context;
        self.made.entry(which).or_insert_with(|| {
            log::debug!("{which:?} compiles on the frame's thread");
            which.compile(context)
        })
    }

    /// A background compile ended.
    pub fn arrived(&mut self, which: BuiltIn, result: Result<wgpu::RenderPipeline, String>) {
        self.compiling.remove(&which);
        match result {
            Ok(pipeline) => {
                self.made.entry(which).or_insert(pipeline);
            }
            Err(message) => log::warn!("{which:?} pipeline: {message}"),
        }
    }
}
