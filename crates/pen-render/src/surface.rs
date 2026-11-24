// The standard surface for material shaders: a base GGX lobe with a clear
// coat, sheen, anisotropy, transmission and a cheap subsurface over it, and
// parallax occlusion mapping, lit by the sky, the sun and the frame's lights.
// The pure lobes (`SURFACE_LOBES_WGSL`) have no bindings, so their tests run
// them in compute shaders; `SURFACE_WGSL` joins them with the sky
// (`ENVIRONMENT_WGSL`) and the clustered lights (`LIGHTS_WGSL`);
// `SURFACE_TRANSMISSION_WGSL` reads the opaque picture behind (it needs the
// translucency library's functions, which declare their own bindings).
// See materials.md.

/// WGSL a material shader adds first to use the standard surface: the
/// `StandardSurface`, its lobes (clear coat, sheen, anisotropy, subsurface,
/// transmission's maths), `surface_light`, `standard_sun`, and `parallax_uv`.
pub const SURFACE_LOBES_WGSL: &str = include_str!("shaders/surface_lobes.wgsl");

/// WGSL that lights a `StandardSurface` with the sky and the frame's lights:
/// `standard_surface`, and `surface_ambient` and `clustered_surface`. It needs
/// `SURFACE_LOBES_WGSL`, `ENVIRONMENT_WGSL` and `LIGHTS_WGSL`.
pub const SURFACE_WGSL: &str = include_str!("shaders/surface.wgsl");

/// WGSL that reads the picture behind a transmissive surface:
/// `surface_transmit`. It needs the two above and the translucency library.
pub const SURFACE_TRANSMISSION_WGSL: &str = include_str!("shaders/surface_transmission.wgsl");

#[cfg(test)]
mod compat;
#[cfg(test)]
mod harness;
#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;
