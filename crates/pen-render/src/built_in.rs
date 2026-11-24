// The renderer's own pipelines, beside its materials' shaders: the passes
// around the scene (post.rs), decals, particles and the inset composite. They compile on
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
    /// Screen-space diffuse global illumination bounce over the opaque scene (ssgi.rs).
    Ssgi(u32),
    /// Temporal antialiasing, reading the depth and velocity of such a scene (taa.rs).
    Taa(u32),
    /// Bokeh depth of field, reading the depth of such a scene (dof.rs).
    Dof(u32),
    /// Camera and per-object velocity motion blur, reading velocity and depth (motion_blur.rs).
    MotionBlur(u32),
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
    /// Particles into a scene of this many samples, added or covering.
    Particles {
        samples: u32,
        additive: bool,
    },
    /// Camera-facing ribbon trails into a scene of this many samples.
    Ribbons {
        samples: u32,
        additive: bool,
    },
    /// The nearest depth of each pixel of a scene of this many samples, for
    /// the translucent surfaces to read.
    OpaqueDepth(u32),
    /// A level of the opaque picture's chain from the one above.
    OpaqueMip,
    /// One axis of the screen-space subsurface blur of a scene of this many samples
    /// (subsurface.rs).
    Subsurface(u32),
    /// The opaque picture laid into every sample of the translucent pass.
    Backdrop(u32),
}

impl BuiltIn {
    /// Every one a frame drawn with `samples` per pixel may draw with.
    pub fn all(samples: u32) -> [BuiltIn; 17] {
        [
            BuiltIn::Occlusion(samples),
            BuiltIn::Atmosphere(samples),
            BuiltIn::Ssgi(samples),
            BuiltIn::Taa(samples),
            BuiltIn::Dof(samples),
            BuiltIn::MotionBlur(samples),
            BuiltIn::Present(samples),
            BuiltIn::BloomDown { first: true },
            BuiltIn::BloomDown { first: false },
            BuiltIn::BloomUp,
            BuiltIn::Decal { samples, paint: true },
            BuiltIn::Decal { samples, paint: false },
            BuiltIn::Composite,
            BuiltIn::OpaqueDepth(samples),
            BuiltIn::OpaqueMip,
            BuiltIn::Backdrop(samples),
            BuiltIn::Subsurface(samples),
        ]
    }

    /// Whether this pipeline depends on the active `DepthMode` (Forward-Z vs Reversed-Z).
    pub fn is_depth_dependent(self) -> bool {
        matches!(
            self,
            BuiltIn::Atmosphere(_)
                | BuiltIn::OpaqueDepth(_)
                | BuiltIn::Particles { .. }
                | BuiltIn::Ribbons { .. }
                | BuiltIn::Decal { .. }
        )
    }

    pub fn compile(self, context: &PipelineContext) -> wgpu::RenderPipeline {
        let clock = crate::platform::Clock::new();
        let pipeline = match self {
            BuiltIn::Decal { samples, paint } => crate::decals::pipeline(context, samples, paint),
            BuiltIn::Composite => crate::renderer::composite_pipeline(context),
            BuiltIn::Particles { samples, additive } => crate::renderer::particle_pipeline(context, samples, additive),
            BuiltIn::Ribbons { samples, additive } => crate::renderer::ribbon_pipeline(context, samples, additive),
            BuiltIn::Backdrop(samples) => crate::post::backdrop_pipeline(context, samples),
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

    /// Clears cached pipelines whose depth comparison or clear value depends on `DepthMode`,
    /// leaving depth-independent pipelines (`Present`, `Composite`, `Backdrop`, etc.) intact
    /// so external bind groups created from their layouts stay valid.
    pub fn invalidate_depth_dependent(&mut self) {
        self.made.retain(|k, _| !k.is_depth_dependent());
        self.compiling.retain(|k| !k.is_depth_dependent());
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
    pub fn arrived(
        &mut self,
        which: BuiltIn,
        depth_mode: crate::camera_precision::DepthMode,
        result: Result<wgpu::RenderPipeline, String>,
    ) {
        if which.is_depth_dependent() && depth_mode != self.context.depth_mode() {
            return;
        }
        self.compiling.remove(&which);
        match result {
            Ok(pipeline) => {
                self.made.entry(which).or_insert(pipeline);
            }
            Err(message) => log::warn!("{which:?} pipeline: {message}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_waits_for_the_translucent_pass_s_pipelines_with_the_rest() {
        for samples in [1, 4] {
            let all = BuiltIn::all(samples);
            for needed in [
                BuiltIn::OpaqueDepth(samples),
                BuiltIn::OpaqueMip,
                BuiltIn::Backdrop(samples),
                BuiltIn::Atmosphere(samples),
                BuiltIn::Subsurface(samples),
            ] {
                assert!(all.contains(&needed), "{needed:?}");
            }
            let distinct: HashSet<BuiltIn> = all.into_iter().collect();
            assert_eq!(distinct.len(), all.len());
        }
    }

    #[test]
    fn particles_and_ribbons_pipelines_compile_cleanly() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return };
        let Ok((device, _queue)) = pollster::block_on(adapter.request_device(&Default::default())) else { return };
        let dummy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[] });
        let dummy_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let ctx = PipelineContext {
            device: device.clone(),
            layout: dummy_pl.clone(),
            skinned_layout: dummy_pl.clone(),
            sprite_layout: dummy_pl,
            scene_group: dummy_layout.clone(),
            draw_group: dummy_layout.clone(),
            skin_group: dummy_layout.clone(),
            lights_group: dummy_layout.clone(),
            sprite_group: dummy_layout,
            format: wgpu::TextureFormat::Bgra8Unorm,
            display: crate::Display::SDR,
            cache: crate::pipeline_cache::PipelineCache::open(&device, None),
            depth_mode: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)),
        };
        for which in [
            BuiltIn::Particles { samples: 1, additive: false },
            BuiltIn::Particles { samples: 1, additive: true },
            BuiltIn::Ribbons { samples: 1, additive: false },
            BuiltIn::Ribbons { samples: 1, additive: true },
        ] {
            let res = pollster::block_on(crate::shader::checked(&device, || which.compile(&ctx)));
            assert!(res.is_ok(), "{which:?} failed: {}", res.unwrap_err());
        }
    }
}
