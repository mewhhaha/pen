//! What the GPU unit tests share: a headless device, half floats, and reading a
//! buffer back.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "bit twiddling of half floats"
)]
use wgpu::util::DeviceExt;

pub fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    pollster::block_on(adapter.request_device(&Default::default())).ok()
}

/// `value` as IEEE half-float bits, rounded to nearest even (finite values within
/// the half range).
pub fn half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return sign;
    }
    assert!(magnitude < 65520.0, "{value} is past half range");
    if magnitude < 6.103_515_6e-5 {
        // Subnormal halves are multiples of 2^-24.
        return sign | (magnitude * 16_777_216.0).round_ties_even() as u16;
    }
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = bits & 0x7f_ffff;
    // Ten bits of mantissa, rounding ties to even; a carry bumps the exponent.
    let kept = mantissa >> 13;
    let rest = mantissa & 0x1fff;
    let mut rounded = ((exponent + 15) as u32) << 10 | kept;
    if rest > 0x1000 || (rest == 0x1000 && kept & 1 == 1) {
        rounded += 1;
    }
    sign | rounded as u16
}

/// The value of half-float bits.
pub fn unhalf(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// An `Rgba16Float` texture of `width` by `height` holding `pixels` (row by row,
/// each four floats).
pub fn picture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    pixels: &[[f32; 4]],
) -> wgpu::Texture {
    assert_eq!(pixels.len(), (width * height) as usize);
    let bits: Vec<u16> = pixels.iter().flatten().map(|&value| half(value)).collect();
    device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("test picture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&bits),
    )
}

/// The first `size` bytes of `buffer` (which can be copied from), once the GPU is
/// done with it.
pub fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let copy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test read"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &copy, 0, size);
    queue.submit([encoder.finish()]);
    copy.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = copy.slice(..).get_mapped_range().unwrap();
    mapped.to_vec()
}
