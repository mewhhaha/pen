# Display scale

How pen handles HiDPI and fractional scaling (125%, 150%, 200%), audited on
2026-10-16. The rule is the browser's: the game lives in physical (device)
pixels. Its viewport is the size it draws at, pointer positions are on that
drawing, and the interface's bounds are in the same pixels, so a game laid
out against its viewport looks the same at every scale and stays sharp.
`pen_runtime::scale` holds the arithmetic both hosts share, unit tested at
scales 1, 1.25, 1.5 and 2.

| What | Native (winit) | Browser |
|---|---|---|
| A new window's or canvas's size | 960 by 640 logical pixels: 1440 by 960 at 150% | The canvas's CSS size times `devicePixelRatio` (`scale::physical_size`, rounded, within the GPU's texture limit) |
| The viewport, the frame | The window's physical size (its swapchain); headless, `--size` | The canvas's backing size |
| Pointer positions | Physical window pixels; when a recording pins the frame's size (`--size` with `--record`), mapped onto the frame stretched over the window (`scale::to_frame`) | CSS pixels mapped onto the backing (`scale::to_frame`), locked motion too |
| Wheel deltas | Lines times 100; pixel deltas (touchpads) divided by the scale factor (`scale::wheel_pixels`), as the browser gives CSS pixels | `deltaY`, CSS pixels |
| Accessibility bounds | Frame pixels times the frame's stretch over the window (`scale::frame_to_window`), physical window pixels as AccessKit wants; macOS's adapter divides by the backing scale itself | The browser's own tree |
| The dev tools | Laid out over the window's physical size; they see window events in window pixels | The canvas's |
| A scale change | winit's `ScaleFactorChanged`, then `Resized` with the new physical size, which resizes the swapchain | The canvas is fitted again each frame |
| macOS | The app bundle's `NSHighResolutionCapable`, without which a Retina display shows a half-size frame scaled up | |

Two fixes came of the audit: a recording's pinned frame took window pixels
as frame pixels (wrong whenever the compositor gave the window another
size, as fractional scales and tiling do), and touchpad scrolling moved
the game twice as far at 200% natively as in the browser.

## Checked headlessly

Without a display: a nested sway with the headless backend
(`WLR_BACKENDS=headless WLR_RENDERER=pixman`, its own `XDG_RUNTIME_DIR`), an
output of 1920x1080 at scale 1.5 and floating windows, and pen on lavapipe
(`--software`) in it:

- Wayland (fractional-scale-v1): the 960x640 logical window drew
  1440x960 frames; the interface's first frame came at 960x640 before the
  compositor said the scale, and every bound in `PEN_A11Y_DUMP` grew by
  exactly 1.5 with it.
- X11 through that sway's Xwayland, `WINIT_X11_SCALE_FACTOR=1.25`: frames
  1200 wide (960 times 1.25; 720 high, the X screen's height), bounds times
  1.25.

Headless runs (`--headless`) have no window and no scale: they draw at
`--size`.
