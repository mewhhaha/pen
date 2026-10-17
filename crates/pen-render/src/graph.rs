// A small render graph, which the passes after the scene draw with (post.rs).
// Each pass is a full-screen triangle drawn by one of the renderer's own
// pipelines: it names the textures it reads, as its bind group's bindings,
// and the one it draws into, or the frame's output.
//
// `plan` checks that every texture is drawn before it is read, leaves out
// the passes whose pictures nothing reads, and gives each texture the graph
// makes a lifetime, from the pass that first draws it to the last that reads
// it. Textures of one size and format whose lifetimes do not overlap share an
// allocation. `compile` makes the allocations and each pass's bind group,
// once for as long as the frames' size and passes stay the same, and
// `execute` encodes the passes every frame.
use anyhow::{Result, ensure};

use crate::built_in::{BuiltIn, BuiltIns};
use crate::renderer::{GpuTimer, Span};

/// A texture a pass reads or draws into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Handle(usize);

/// A texture the graph makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Spec {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

enum Source {
    /// Made by the graph, perhaps sharing an allocation with another.
    Made(Spec),
    /// Made elsewhere: the scene's targets, a history kept between frames.
    /// Passes drawing into one always draw.
    Imported(wgpu::TextureView),
}

struct Texture {
    name: &'static str,
    source: Source,
}

/// One of a pass's bindings, in binding order from 0.
#[derive(Clone)]
pub(crate) enum Binding {
    Uniform(wgpu::Buffer),
    Texture(Handle),
    Sampler(wgpu::Sampler),
}

/// Where a pass draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    Texture(Handle),
    /// The frame's output, which `execute` is given.
    Output,
}

struct Pass {
    name: &'static str,
    pipeline: BuiltIn,
    bindings: Vec<Binding>,
    target: Target,
    /// Whether it draws over what its target holds, blending into it, rather
    /// than clearing it first.
    over: bool,
}

impl Pass {
    fn reads(&self) -> impl Iterator<Item = Handle> + '_ {
        self.bindings.iter().filter_map(|binding| match binding {
            Binding::Texture(handle) => Some(*handle),
            _ => None,
        })
    }
}

/// The passes of a graph and their textures, in the order they draw.
#[derive(Default)]
pub(crate) struct GraphBuilder {
    textures: Vec<Texture>,
    passes: Vec<Pass>,
}

/// What `plan` works out: the passes that draw, in order, and which
/// allocation each texture the graph makes is in.
#[derive(Debug, PartialEq)]
struct Plan {
    passes: Vec<usize>,
    /// By texture: its allocation, if the graph makes it and a pass uses it.
    allocation: Vec<Option<usize>>,
    allocations: Vec<Spec>,
}

impl GraphBuilder {
    /// A texture made elsewhere, read or drawn into through `view`.
    pub fn import(&mut self, name: &'static str, view: wgpu::TextureView) -> Handle {
        self.textures.push(Texture { name, source: Source::Imported(view) });
        Handle(self.textures.len() - 1)
    }

    /// A texture the graph makes, for passes to draw into and read.
    pub fn texture(&mut self, name: &'static str, spec: Spec) -> Handle {
        self.textures.push(Texture { name, source: Source::Made(spec) });
        Handle(self.textures.len() - 1)
    }

    /// A pass drawing `pipeline` with `bindings` into `target`, clearing it
    /// first or, `over`, blending over what it holds.
    pub fn pass(&mut self, name: &'static str, pipeline: BuiltIn, bindings: Vec<Binding>, target: Target, over: bool) {
        self.passes.push(Pass { name, pipeline, bindings, target, over });
    }

    fn imported(&self, handle: Handle) -> bool {
        matches!(self.textures[handle.0].source, Source::Imported(_))
    }

    fn plan(&self) -> Result<Plan> {
        // What each texture holds is drawn before anything reads it.
        let mut drawn: Vec<bool> = (0..self.textures.len()).map(|index| self.imported(Handle(index))).collect();
        for pass in &self.passes {
            for read in pass.reads() {
                ensure!(drawn[read.0], "{} reads {} before a pass draws it", pass.name, self.textures[read.0].name);
            }
            if let Target::Texture(target) = pass.target {
                let name = self.textures[target.0].name;
                ensure!(pass.reads().all(|read| read != target), "{} reads {name}, which it draws into", pass.name);
                ensure!(!pass.over || drawn[target.0], "{} draws over {name} before a pass draws it", pass.name);
                drawn[target.0] = true;
            }
        }
        // From the last pass back: those whose pictures the output, a texture
        // made elsewhere, or a later pass that draws reads.
        let mut needed = vec![false; self.textures.len()];
        let mut live = vec![false; self.passes.len()];
        for (index, pass) in self.passes.iter().enumerate().rev() {
            let target = match pass.target {
                Target::Output => None,
                Target::Texture(target) => Some(target),
            };
            if target.is_some_and(|target| !needed[target.0] && !self.imported(target)) {
                continue;
            }
            live[index] = true;
            if let Some(target) = target {
                // Drawing over it needs what was there.
                needed[target.0] = pass.over;
            }
            for read in pass.reads() {
                needed[read.0] = true;
            }
        }
        let passes: Vec<usize> = (0..self.passes.len()).filter(|&index| live[index]).collect();
        // Each texture's first and last use, by place in `passes`.
        let mut uses: Vec<Option<(usize, usize)>> = vec![None; self.textures.len()];
        for (place, &index) in passes.iter().enumerate() {
            let pass = &self.passes[index];
            let target = match pass.target {
                Target::Texture(target) => Some(target),
                Target::Output => None,
            };
            for handle in pass.reads().chain(target) {
                let span = uses[handle.0].get_or_insert((place, place));
                span.1 = place;
            }
        }
        // First used first, each into an allocation of its spec that is free
        // by then, or a new one.
        let mut order: Vec<usize> = (0..self.textures.len()).filter(|&index| uses[index].is_some()).collect();
        order.sort_by_key(|&index| uses[index].unwrap().0);
        let mut allocation = vec![None; self.textures.len()];
        let mut allocations: Vec<Spec> = Vec::new();
        let mut free_after: Vec<usize> = Vec::new();
        for index in order {
            let Source::Made(spec) = self.textures[index].source else { continue };
            let (first, last) = uses[index].unwrap();
            let shared = (0..allocations.len()).find(|&slot| allocations[slot] == spec && free_after[slot] < first);
            let slot = shared.unwrap_or_else(|| {
                allocations.push(spec);
                free_after.push(0);
                allocations.len() - 1
            });
            free_after[slot] = last;
            allocation[index] = Some(slot);
        }
        Ok(Plan { passes, allocation, allocations })
    }

    /// Makes the textures and each pass's bind group from its pipeline. A
    /// graph drawn in other frames than `sharing`, with the same passes but
    /// other imports (a history read in turns), shares its textures.
    pub fn compile(self, device: &wgpu::Device, built_ins: &mut BuiltIns, sharing: Option<&Graph>) -> Result<Graph> {
        let plan = self.plan()?;
        if let Some(other) = sharing {
            ensure!(other.specs == plan.allocations, "a graph sharing textures needs the same ones");
        }
        let made: Vec<wgpu::Texture> = match sharing {
            Some(other) => other.allocations.clone(),
            None => plan
                .allocations
                .iter()
                .map(|spec| {
                    device.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        size: wgpu::Extent3d { width: spec.width, height: spec.height, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: spec.format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                })
                .collect(),
        };
        let views: Vec<Option<wgpu::TextureView>> = self
            .textures
            .iter()
            .zip(&plan.allocation)
            .map(|(texture, allocation)| match &texture.source {
                Source::Imported(view) => Some(view.clone()),
                Source::Made(_) => allocation.map(|slot| made[slot].create_view(&Default::default())),
            })
            .collect();
        let view = |handle: Handle| views[handle.0].as_ref().expect("a used texture has a view");
        let mut passes = Vec::with_capacity(plan.passes.len());
        for index in plan.passes {
            let pass = &self.passes[index];
            let pipeline = built_ins.get(pass.pipeline);
            let entries: Vec<wgpu::BindGroupEntry<'_>> = (0..)
                .zip(&pass.bindings)
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding,
                    resource: match resource {
                        Binding::Uniform(buffer) => buffer.as_entire_binding(),
                        Binding::Texture(handle) => wgpu::BindingResource::TextureView(view(*handle)),
                        Binding::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                    },
                })
                .collect();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(pass.name),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let target = match pass.target {
                Target::Texture(handle) => Some(view(handle).clone()),
                Target::Output => None,
            };
            passes.push(CompiledPass { name: pass.name, pipeline: pass.pipeline, group, target, over: pass.over });
        }
        Ok(Graph { passes, specs: plan.allocations, allocations: made })
    }
}

struct CompiledPass {
    name: &'static str,
    pipeline: BuiltIn,
    group: wgpu::BindGroup,
    /// None for the output.
    target: Option<wgpu::TextureView>,
    over: bool,
}

/// A compiled graph: its passes in order, with their bind groups and the
/// textures it made.
pub(crate) struct Graph {
    passes: Vec<CompiledPass>,
    specs: Vec<Spec>,
    allocations: Vec<wgpu::Texture>,
}

impl Graph {
    /// Encodes every pass, the last ones onto `output`, each timed as one
    /// of the passes after the scene where `timer` times the frame.
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        built_ins: &mut BuiltIns,
        output: &wgpu::TextureView,
        timer: &GpuTimer,
    ) {
        for pass in &self.passes {
            let mut drawing = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(pass.name),
                timestamp_writes: timer.pass(Span::Post),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: pass.target.as_ref().unwrap_or(output),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if pass.over { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(wgpu::Color::BLACK) },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            drawing.set_pipeline(built_ins.get(pass.pipeline));
            drawing.set_bind_group(0, &pass.group, &[]);
            drawing.draw(0..3, 0..1);
        }
    }

    /// The passes that draw, in order.
    pub fn names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|pass| pass.name).collect()
    }

    /// How many textures it made.
    pub fn allocations(&self) -> usize {
        self.allocations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Plans need no device: imported views are never touched, so the tests
    // import none and mark the output's passes instead.
    fn spec(width: u32) -> Spec {
        Spec { width, height: width, format: wgpu::TextureFormat::Rgba16Float }
    }

    fn pass(graph: &mut GraphBuilder, name: &'static str, reads: &[Handle], target: Target, over: bool) {
        let bindings = reads.iter().map(|&handle| Binding::Texture(handle)).collect();
        graph.pass(name, BuiltIn::BloomUp, bindings, target, over);
    }

    fn names(graph: &GraphBuilder, plan: &Plan) -> Vec<&'static str> {
        plan.passes.iter().map(|&index| graph.passes[index].name).collect()
    }

    #[test]
    fn leaves_out_what_nothing_reads_and_shares_allocations() {
        let mut graph = GraphBuilder::default();
        let a = graph.texture("a", spec(8));
        let b = graph.texture("b", spec(8));
        let unread = graph.texture("unread", spec(8));
        let c = graph.texture("c", spec(8));
        let small = graph.texture("small", spec(4));
        pass(&mut graph, "draw a", &[], Target::Texture(a), false);
        pass(&mut graph, "a into b", &[a], Target::Texture(b), false);
        pass(&mut graph, "a into unread", &[a], Target::Texture(unread), false);
        pass(&mut graph, "b into small", &[b], Target::Texture(small), false);
        // a's last read is done: c can take its allocation, not b's.
        pass(&mut graph, "small into c", &[small], Target::Texture(c), false);
        pass(&mut graph, "present", &[b, c], Target::Output, false);
        let plan = graph.plan().unwrap();
        assert_eq!(names(&graph, &plan), ["draw a", "a into b", "b into small", "small into c", "present"]);
        assert_eq!(plan.allocation, [Some(0), Some(1), None, Some(0), Some(2)]);
        assert_eq!(plan.allocations, [spec(8), spec(8), spec(4)]);
    }

    #[test]
    fn drawing_over_keeps_what_it_draws_over() {
        let mut graph = GraphBuilder::default();
        let low = graph.texture("low", spec(4));
        let high = graph.texture("high", spec(8));
        let lost = graph.texture("lost", spec(8));
        pass(&mut graph, "down", &[], Target::Texture(high), false);
        pass(&mut graph, "down again", &[high], Target::Texture(low), false);
        pass(&mut graph, "up", &[low], Target::Texture(high), true);
        // Cleared by the next pass before anything reads it.
        pass(&mut graph, "overwritten", &[], Target::Texture(lost), false);
        pass(&mut graph, "redrawn", &[high], Target::Texture(lost), false);
        pass(&mut graph, "present", &[lost], Target::Output, false);
        let plan = graph.plan().unwrap();
        assert_eq!(names(&graph, &plan), ["down", "down again", "up", "redrawn", "present"]);
        // high lives from "down" to "redrawn", so lost cannot share it.
        assert_eq!(plan.allocation, [Some(1), Some(0), Some(2)]);
    }

    #[test]
    fn refuses_reads_of_what_nothing_drew() {
        let mut graph = GraphBuilder::default();
        let a = graph.texture("a", spec(8));
        pass(&mut graph, "present", &[a], Target::Output, false);
        assert_eq!(graph.plan().unwrap_err().to_string(), "present reads a before a pass draws it");
        let mut graph = GraphBuilder::default();
        let a = graph.texture("a", spec(8));
        pass(&mut graph, "blend", &[], Target::Texture(a), true);
        assert!(graph.plan().is_err());
        let mut graph = GraphBuilder::default();
        let a = graph.texture("a", spec(8));
        pass(&mut graph, "draw", &[], Target::Texture(a), false);
        pass(&mut graph, "feedback", &[a], Target::Texture(a), false);
        assert_eq!(graph.plan().unwrap_err().to_string(), "feedback reads a, which it draws into");
    }
}
