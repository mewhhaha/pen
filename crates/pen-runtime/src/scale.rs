//! Display scale, as both hosts handle it. A window or canvas has a size in
//! logical pixels (CSS pixels in the browser) and is drawn in physical
//! (device) pixels: the logical size times the display's scale, 1, 1.25,
//! 1.5, 2 and so on. The game lives in physical pixels. Its viewport is the
//! size it draws at, pointer positions are on that drawing, and the bounds
//! of the interface nodes it declares are in them too. Wheel deltas are the
//! exception: the browser gives them in CSS pixels, so a native pixel delta
//! is divided down to match, and a scroll moves the game as far on either.

/// `logical` at `scale` in physical pixels, rounded, each side from 1 to
/// `limit` (the largest texture the GPU takes).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "within 1..=limit first")]
pub fn physical_size(logical: (f64, f64), scale: f64, limit: u32) -> (u32, u32) {
    // max before min: NaN becomes 1.
    let side = |length: f64| (length * scale).round().max(1.0).min(f64::from(limit.max(1))) as u32;
    (side(logical.0), side(logical.1))
}

/// Where `position`, in the pixels of a surface `shown` wide and high,
/// falls on a frame of `frame` pixels stretched over it: a pointer over a
/// canvas whose backing differs from its CSS size, or over a window showing
/// frames of a pinned size (a recording's). An empty surface maps nothing.
pub fn to_frame(position: (f64, f64), shown: (f64, f64), frame: (u32, u32)) -> (f64, f64) {
    let axis = |at: f64, shown: f64, frame: u32| if shown > 0.0 { at * f64::from(frame) / shown } else { at };
    (axis(position.0, shown.0, frame.0), axis(position.1, shown.1, frame.1))
}

/// How far a frame of `frame` pixels is stretched to fill a window of
/// `window` pixels, on each axis: what turns the interface's bounds, in the
/// frame's pixels, into the window's for assistive technology. 1 when the
/// window draws at its own size.
pub fn frame_to_window(window: (u32, u32), frame: (u32, u32)) -> [f64; 2] {
    [f64::from(window.0) / f64::from(frame.0.max(1)), f64::from(window.1) / f64::from(frame.1.max(1))]
}

/// A wheel's delta in physical pixels (a touchpad's, on Wayland or macOS) as
/// the CSS pixels a browser's `WheelEvent.deltaY` gives.
pub fn wheel_pixels(physical: f64, scale: f64) -> f64 {
    if scale > 0.0 { physical / scale } else { physical }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scales displays use: whole, and the fractional ones.
    const SCALES: [f64; 4] = [1.0, 1.25, 1.5, 2.0];

    #[test]
    fn logical_sizes_become_device_pixels() {
        let expected = [(960, 640), (1200, 800), (1440, 960), (1920, 1280)];
        for (scale, expected) in SCALES.into_iter().zip(expected) {
            assert_eq!(physical_size((960.0, 640.0), scale, 8192), expected, "{scale}");
        }
        // A side that falls between pixels rounds; 1366 x 1.25 = 1707.5.
        assert_eq!(physical_size((1366.0, 768.0), 1.25, 8192), (1708, 960));
        assert_eq!(physical_size((1366.0, 768.0), 1.5, 8192), (2049, 1152));
        // Never empty, never past the GPU's limit, whatever the page says.
        assert_eq!(physical_size((0.0, 0.4), 2.0, 8192), (1, 1));
        assert_eq!(physical_size((10_000.0, 300.0), 2.0, 8192), (8192, 600));
        assert_eq!(physical_size((f64::NAN, -5.0), 1.5, 8192), (1, 1));
    }

    #[test]
    fn a_window_drawn_at_its_own_size_maps_positions_to_themselves() {
        for scale in SCALES {
            let window = physical_size((960.0, 640.0), scale, 8192);
            let shown = (f64::from(window.0), f64::from(window.1));
            for position in [(0.0, 0.0), (123.5, 77.25), (shown.0, shown.1)] {
                assert_eq!(to_frame(position, shown, window), position, "{scale}");
            }
            assert_eq!(frame_to_window(window, window), [1.0, 1.0]);
        }
    }

    #[test]
    fn a_canvas_maps_css_pixels_to_its_backing() {
        // The browser: a canvas 480 by 320 CSS pixels, backed by device pixels.
        for scale in SCALES {
            let backing = physical_size((480.0, 320.0), scale, 8192);
            let (x, y) = to_frame((240.0, 160.0), (480.0, 320.0), backing);
            assert_eq!((x, y), (240.0 * scale, 160.0 * scale), "{scale}");
            let corner = to_frame((480.0, 320.0), (480.0, 320.0), backing);
            assert_eq!(corner, (f64::from(backing.0), f64::from(backing.1)));
        }
    }

    #[test]
    fn a_pinned_frame_maps_through_the_window_that_shows_it() {
        // A 1920x1080 recording in a window the compositor made smaller: at
        // each scale, a window of 1280x720 logical pixels.
        let frame = (1920, 1080);
        for scale in SCALES {
            let window = physical_size((1280.0, 720.0), scale, 8192);
            let shown = (f64::from(window.0), f64::from(window.1));
            assert_eq!(to_frame((shown.0 / 2.0, shown.1 / 2.0), shown, frame), (960.0, 540.0), "{scale}");
            assert_eq!(to_frame(shown, shown, frame), (1920.0, 1080.0));
            // Accessibility goes the other way: frame pixels to the window's.
            let [x, y] = frame_to_window(window, frame);
            let back = to_frame((960.0 * x, 540.0 * y), shown, frame);
            assert!((back.0 - 960.0).abs() < 1e-9 && (back.1 - 540.0).abs() < 1e-9, "{scale}: {back:?}");
        }
        // A minimized window maps nothing rather than dividing by zero.
        assert_eq!(to_frame((5.0, 6.0), (0.0, 0.0), frame), (5.0, 6.0));
        assert_eq!(frame_to_window((0, 0), (0, 0)), [0.0, 0.0]);
    }

    #[test]
    fn wheels_scroll_as_far_at_every_scale() {
        for scale in SCALES {
            // One notch of 120 CSS pixels, as a touchpad reports it natively.
            assert_eq!(wheel_pixels(120.0 * scale, scale), 120.0, "{scale}");
        }
        assert_eq!(wheel_pixels(50.0, 0.0), 50.0);
    }
}
