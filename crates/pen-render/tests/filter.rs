//! The colour filter on the GPU: the picture the frame draws, through the
//! filter's matrix, as `ColorFilter::apply` works it out.
mod common;

use common::*;
use pen_protocol::{Catalog, Color, color_filter};
use pen_render::{ColorFilter, Renderer, RendererOptions};

/// The middle pixel of a frame cleared to `clear`, drawn through `filter`.
fn drawn(
    (device, queue): &(wgpu::Device, wgpu::Queue),
    renderer: &mut Renderer,
    target: &mut Offscreen,
    clear: [f32; 3],
    filter: ColorFilter,
) -> [u8; 4] {
    let mut frame = frame(Vec::new());
    frame.clear = Color { r: clear[0], g: clear[1], b: clear[2], a: 1.0 };
    renderer.set_color_filter(filter);
    // The passes around the scene compile in the background first.
    for _ in 0..500 {
        renderer.submit(&frame, &Catalog::default(), target).unwrap();
        if renderer.pending() == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    renderer.submit(&frame, &Catalog::default(), target).unwrap();
    middle(device, queue, &target.0)
}

#[allow(clippy::cast_possible_truncation, reason = "a channel, 0 to 255")]
fn byte(value: f32) -> i32 {
    (value.clamp(0.0, 1.0) * 255.0).round() as i32
}

#[test]
fn filters_the_whole_picture() {
    let Some(gpu) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let assets = std::env::temp_dir().join(format!("pen-filter-test-{}", std::process::id()));
    let options =
        RendererOptions { format: FORMAT, display: pen_render::Display::SDR, assets, veil: None, pipeline_cache: None };
    let mut renderer = Renderer::new(gpu.0.clone(), gpu.1.clone(), options).unwrap();
    let mut target = Offscreen::new(&gpu.0, 64);
    let clear = [0.8, 0.25, 0.2];
    let plain = drawn(&gpu, &mut renderer, &mut target, clear, ColorFilter::default());
    let near = |shown: [u8; 4], expected: [i32; 3], within: i32| {
        (0..3).all(|channel| (i32::from(shown[channel]) - expected[channel]).abs() <= within)
    };
    assert!(near(plain, clear.map(byte), 1), "{plain:?}");
    let kinds =
        [color_filter::PROTANOPIA, color_filter::DEUTERANOPIA, color_filter::TRITANOPIA, color_filter::ACHROMATOPSIA];
    for kind in kinds {
        for simulate in [false, true] {
            let filter = ColorFilter { kind, strength: 1.0, simulate };
            let shown = drawn(&gpu, &mut renderer, &mut target, clear, filter);
            let expected = filter.apply(clear).map(byte);
            assert!(near(shown, expected, 2), "{kind} {simulate}: {shown:?} against {expected:?}");
            assert_ne!(shown[..3], plain[..3], "{kind} {simulate} changed nothing");
        }
    }
    // Off again, the picture is as it was.
    assert_eq!(drawn(&gpu, &mut renderer, &mut target, clear, ColorFilter::default()), plain);
}
