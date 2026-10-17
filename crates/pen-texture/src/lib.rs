//! Textures as pen ships them: KTX2 files (`ktx2`), one per format family
//! a GPU may sample directly (BC7, ASTC, ETC2, or plain RGBA). The renderer
//! reads them; with the `tools` feature the asset pipeline also writes them,
//! with the block encoders and mip chains that takes, and makes multi-channel
//! signed distance field (MSDF) atlases of fonts.
pub mod ktx2;

#[cfg(feature = "tools")]
pub mod astc;
#[cfg(feature = "tools")]
pub mod bc7;
#[cfg(feature = "tools")]
pub mod encode;
#[cfg(feature = "tools")]
pub mod etc2;
#[cfg(feature = "tools")]
pub mod mips;
#[cfg(feature = "tools")]
pub mod msdf;
