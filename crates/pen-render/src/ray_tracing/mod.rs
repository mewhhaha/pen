//! Portable GPU BVH Ray-Traced Soft Shadows & Ray-Traced Ambient Occlusion (RTAO),
//! plus hardware `EXPERIMENTAL_RAY_QUERY` capability detection.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU BVH node and texture indices")]

mod bvh;
mod pool;
mod tracer;

#[cfg(test)]
mod tests;

pub use bvh::{
    Blas, BlasTriGpu, BlasTriangle, BvhNodeGpu, BvhScene, BvhTriGpu, BvhTriangle, TlasInstance, TlasInstanceGpu,
    TlasScene, TlasSceneGpu,
};
pub use pool::GpuBlasPool;
pub use tracer::{GpuRayTracer, MAX_RT_LOCAL_LIGHTS, RayTraceParams, RtLocalLight, TlasRayTraceParams, TlasRayTracer};

pub const BVH_RAYTRACE_WGSL: &str = include_str!("../shaders/bvh_raytrace.wgsl");
pub const TLAS_TRAVERSAL_WGSL: &str = include_str!("../shaders/tlas_traversal.wgsl");
pub const TLAS_RAYTRACE_WGSL: &str = include_str!("../shaders/tlas_raytrace.wgsl");
#[cfg(not(target_arch = "wasm32"))]
pub const TLAS_TRAVERSAL_HW_WGSL: &str = include_str!("../shaders/tlas_traversal_hw.wgsl");

/// Reports whether the active GPU adapter advertises hardware ray-query support (`KHR_ray_query` / Metal RT / DXR).
/// Note that [`GpuRayTracer`] executes portably via compute BVH traversal on ALL WebGPU/Vulkan/Metal backends.
pub fn supports_hardware_ray_query(adapter: &wgpu::Adapter) -> bool {
    adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
}

/// Reports whether `device` was opened with hardware ray-query enabled AND valid acceleration structure limits.
pub fn device_supports_hardware_ray_query(device: &wgpu::Device) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = device;
        false
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let limits = device.limits();
        device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            && limits.max_blas_primitive_count > 0
            && limits.max_tlas_instance_count > 0
            && limits.max_acceleration_structures_per_shader_stage > 0
    }
}

fn encode_copy(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer, bytes: &[u8]) {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt_stage"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    stage.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytes);
    stage.unmap();
    encoder.copy_buffer_to_buffer(&stage, 0, dst, 0, bytes.len() as u64);
}
