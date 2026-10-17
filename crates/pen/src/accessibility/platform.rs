//! Where the tree goes: the window's AccessKit adapter, or, headless on
//! Linux with PEN_A11Y_ATSPI=1, an AT-SPI adapter of its own on the session
//! bus, so that a screen reader (or a test's AT-SPI client, tests/atspi)
//! reads and works a headless run as it would a window. Such a run keeps to
//! the game's own pace, a frame each 1/fps of wall time, so that what
//! assistive technology does reaches the game while the run lasts.
use accesskit::{ActionHandler, ActivationHandler, DeactivationHandler, TreeUpdate};
use winit::event::WindowEvent;
use winit::window::Window;

pub(super) enum Platform {
    Window(accesskit_winit::Adapter),
    /// Headless, on the AT-SPI bus, and the frame size its window bounds have.
    #[cfg(target_os = "linux")]
    Bus(accesskit_unix::Adapter, (u32, u32)),
}

impl Platform {
    pub fn update_if_active(&mut self, update: impl FnOnce() -> TreeUpdate) {
        match self {
            Platform::Window(adapter) => adapter.update_if_active(update),
            #[cfg(target_os = "linux")]
            Platform::Bus(adapter, _) => adapter.update_if_active(update),
        }
    }

    pub fn window_event(&mut self, window: &Window, event: &WindowEvent) {
        if let Platform::Window(adapter) = self {
            adapter.process_event(window, event);
        }
    }

    /// Headless: the frame's size, as the bounds of the window it stands for.
    pub fn frame_size(&mut self, size: (u32, u32)) {
        #[cfg(target_os = "linux")]
        if let Platform::Bus(adapter, bounds) = self
            && *bounds != size
        {
            *bounds = size;
            let rect = accesskit::Rect::new(0.0, 0.0, f64::from(size.0), f64::from(size.1));
            adapter.set_root_window_bounds(rect, rect);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = size;
    }
}

/// Whether a headless run shows its interface on the AT-SPI bus
/// (PEN_A11Y_ATSPI=1, on Linux).
pub fn bus_wanted() -> bool {
    static WANTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WANTED.get_or_init(|| {
        cfg!(target_os = "linux")
            && std::env::var_os("PEN_A11Y_ATSPI").is_some_and(|value| !value.is_empty() && value != "0")
    })
}

/// Headless: an adapter on the AT-SPI bus, when one is wanted. It stands
/// for a window with the keyboard focus, as screen readers follow the focus
/// only in the active window.
pub(super) fn bus(
    activation: impl 'static + ActivationHandler + Send,
    action: impl 'static + ActionHandler + Send,
    deactivation: impl 'static + DeactivationHandler + Send,
) -> Option<Platform> {
    if !bus_wanted() {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let mut adapter = accesskit_unix::Adapter::new(activation, action, deactivation);
        adapter.update_window_focus_state(true);
        Some(Platform::Bus(adapter, (0, 0)))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (activation, action, deactivation);
        None
    }
}
