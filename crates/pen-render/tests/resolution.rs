//! Dynamic resolution on a real (headless) GPU: with a frame budget no GPU
//! can hold, the scene's render scale drops to half the frame's, timed by
//! timestamps where the device has them and by the queue finishing its work
//! where it does not; without a budget it draws at the frame's scale. Skips
//! when no adapter is available.
#![cfg(not(target_arch = "wasm32"))]
mod common;

use common::*;
use pen_protocol::*;
use pen_render::{Renderer, RendererOptions};

#[test]
fn render_scale_drops_to_hold_a_frame_budget() {
    for featured in [false, true] {
        let Some((device, queue)) = (if featured { featured_device() } else { device() }) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pen-render-budget-{featured}"));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("catalog.json"), br#"{"meshes":[],"materials":[]}"#).unwrap();
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererOptions {
                format: FORMAT,
                display: pen_render::Display::SDR,
                assets: root,
                veil: None,
                pipeline_cache: None,
            },
        )
        .unwrap();
        let catalog = Catalog::default();
        let mut target = Offscreen::new(&device, 64);
        let mut frame = frame(Vec::new());
        frame.quality.scale = 0.8;
        let mut run = |renderer: &mut Renderer, frames: usize| {
            for _ in 0..frames {
                renderer.submit(&frame, &catalog, &mut target).unwrap();
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            }
            renderer.stats().scale
        };
        // The scale as drawn, in whole pixels of 64.
        let near = |scale: f32, wanted: f32| (scale - wanted).abs() < 0.02;
        assert!(near(run(&mut renderer, 3), 0.8));
        // A nanosecond: down to half the frame's scale within a few holds.
        renderer.frame_budget(Some(1e-6));
        let scale = run(&mut renderer, 200);
        assert!(near(scale, 0.4), "timestamps {featured}: scale {scale}");
        renderer.frame_budget(None);
        assert!(near(run(&mut renderer, 2), 0.8));
    }
}
