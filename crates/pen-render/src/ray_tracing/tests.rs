use super::{
    Blas, BlasTriangle, BvhScene, BvhTriangle, GpuBlasPool, GpuRayTracer, RayTraceParams, TlasInstance,
    TlasRayTraceParams, TlasRayTracer, TlasScene, supports_hardware_ray_query,
};

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let frac = u32::from(bits & 0x3ff);
    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign)
        } else {
            let mut f = frac;
            let mut e = -14i32;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            f32::from_bits(sign | (((e + 127) as u32) << 23) | (f << 13))
        }
    } else if exp == 31 {
        f32::from_bits(sign | (0xff << 23) | (frac << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
    }
}

#[test]
fn gpu_bvh_watertight_raytraced_shadow_and_concave_rtao_witness() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("GPU adapter required");
    let _hw_rq = supports_hardware_ray_query(&adapter);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

    let tris = [
        // Horizontal occluder roof quad at y = 1.5 over x in [-1.8, -0.5] (shared diagonal tests watertightness!).
        BvhTriangle { v0: [-1.8, 1.5, -1.5], v1: [-0.5, 1.5, -1.5], v2: [-0.5, 1.5, 1.5] },
        BvhTriangle { v0: [-1.8, 1.5, -1.5], v1: [-0.5, 1.5, 1.5], v2: [-1.8, 1.5, 1.5] },
        // Vertical wall at x = -0.5, y in [0, 2.5], z in [-2, 2] for RTAO concavity!
        BvhTriangle { v0: [-0.5, 0.0, -2.0], v1: [-0.5, 2.5, -2.0], v2: [-0.5, 2.5, 2.0] },
        BvhTriangle { v0: [-0.5, 0.0, -2.0], v1: [-0.5, 2.5, 2.0], v2: [-0.5, 0.0, 2.0] },
        // Degenerate zero-area triangle must be filtered cleanly without breaking build.
        BvhTriangle { v0: [0.0, 0.0, 0.0], v1: [0.0, 0.0, 0.0], v2: [0.0, 0.0, 0.0] },
    ];
    let bvh = BvhScene::build(&tris).unwrap();
    assert_eq!(bvh.tris().len(), 4, "zero-area degenerate triangle must be filtered");
    let mut rt = GpuRayTracer::new(&device);
    rt.upload_bvh(&device, &bvh).unwrap();

    let vp: [f32; 16] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0];
    let (w, h) = (32u32, 32u32);
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rt_depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &depth_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&vec![0.5f32; (w * h) as usize]),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    rt.trace(
        &device,
        &mut enc,
        &depth_view,
        (w, h),
        &RayTraceParams {
            view_projection: vp,
            camera_pos: [0.0, 4.0, 0.0],
            depth_clear: 1.0,
            sun_direction: [0.0, 1.0, 0.0], // Axis-aligned ray (+Y) tests parallel X/Z slab handling!
            sun_angular_radius: 0.0,
            ao_radius: 2.5,
            ao_samples: 16,
            shadow_samples: 1,
            frame_seed: 1.0,
            normal_bias: 0.01,
        },
    )
    .unwrap();

    let bpr = (w * 8).next_multiple_of(256);
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt_rb"),
        size: u64::from(bpr * h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: rt.output_texture().unwrap(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &rb,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(Some(enc.finish()));
    let slice = rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range().unwrap();
    let read_px = |x: u32, y: u32| -> [f32; 4] {
        let o = (y * bpr + x * 8) as usize;
        [
            f16_to_f32(u16::from_le_bytes([data[o], data[o + 1]])),
            f16_to_f32(u16::from_le_bytes([data[o + 2], data[o + 3]])),
            f16_to_f32(u16::from_le_bytes([data[o + 4], data[o + 5]])),
            f16_to_f32(u16::from_le_bytes([data[o + 6], data[o + 7]])),
        ]
    };
    let under_roof = read_px(7, 16);
    let open_floor = read_px(26, 16);
    assert!(under_roof[0] < 0.05, "occluder roof must cast exact dark RT shadow: {under_roof:?}");
    assert!(open_floor[0] > 0.95, "open floor must stay fully lit: {open_floor:?}");
    assert!(
        under_roof[1] + 0.15 < open_floor[1],
        "concave wall+roof corner must have darker RTAO than open floor: corner={} open={}",
        under_roof[1],
        open_floor[1]
    );
    drop(data);
    rb.unmap();
}

fn read_rgba16f(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, w: u32, h: u32) -> Vec<[f32; 4]> {
    let bpr = (w * 8).next_multiple_of(256);
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tlas_rb"),
        size: u64::from(bpr * h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &rb,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(Some(enc.finish()));
    let slice = rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let o = (y * bpr + x * 8) as usize;
            out.push([
                f16_to_f32(u16::from_le_bytes([data[o], data[o + 1]])),
                f16_to_f32(u16::from_le_bytes([data[o + 2], data[o + 3]])),
                f16_to_f32(u16::from_le_bytes([data[o + 4], data[o + 5]])),
                f16_to_f32(u16::from_le_bytes([data[o + 6], data[o + 7]])),
            ]);
        }
    }
    drop(data);
    rb.unmap();
    out
}

#[test]
fn gpu_tlas_blas_dynamic_transform_refit_reflection_and_gi_witness() {
    let (device, queue) = crate::testing::device().expect("GPU adapter required");

    // Reusable unit quad BLAS in XZ plane centered at origin ([-0.5..0.5] x 0 x [-1.5..1.5]).
    let quad_tris = [
        BlasTriangle::flat([-0.6, 0.0, -1.5], [0.6, 0.0, -1.5], [0.6, 0.0, 1.5]),
        BlasTriangle::flat([-0.6, 0.0, -1.5], [0.6, 0.0, 1.5], [-0.6, 0.0, 1.5]),
    ];
    let mut roof_blas = Blas::build(&quad_tris).unwrap();

    let translate = |x: f32, y: f32, z: f32| -> [f32; 16] {
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0]
    };

    // Frame 1: glowing red roof over LEFT half (x = -1.15, y = 1.5).
    let mut left_inst = TlasInstance::new(0, translate(-1.15, 1.5, 0.0));
    left_inst.albedo = [0.9, 0.05, 0.05];
    left_inst.emissive = [3.0, 0.0, 0.0];
    let scene1 = TlasScene::build(&[roof_blas.clone()], &[left_inst]).unwrap();

    let mut rt = TlasRayTracer::new(&device);
    rt.upload_scene(&device, &scene1).unwrap();

    let vp: [f32; 16] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0];
    let (w, h) = (32u32, 32u32);
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tlas_depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        depth_tex.as_image_copy(),
        bytemuck::cast_slice(&vec![0.5f32; (w * h) as usize]),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let depth_view = depth_tex.create_view(&Default::default());

    let params = TlasRayTraceParams {
        view_projection: vp,
        camera_pos: [0.0, 4.0, 0.0],
        depth_clear: 1.0,
        sun_direction: [0.0, 1.0, 0.0],
        sun_angular_radius: 0.0,
        sun_color: [2.0, 2.0, 2.0],
        sky_zenith: [0.05, 0.1, 0.3],
        sky_horizon: [0.05, 0.1, 0.25],
        surface_roughness: 0.0, // mirror specular reflection directly upward to hit red roof
        ao_radius: 3.0,
        max_ray_distance: 100.0,
        shadow_samples: 1,
        ao_samples: 8,
        reflection_samples: 1,
        gi_samples: 8,
        gi_bounce_scale: 1.0,
        frame_seed: 1.0,
        normal_bias: 0.01,
        visibility_mask: u32::MAX,
    };

    let mut enc = device.create_command_encoder(&Default::default());
    rt.trace(&device, &mut enc, &depth_view, (w, h), &params).unwrap();
    queue.submit(Some(enc.finish()));

    let sh1 = read_rgba16f(&device, &queue, rt.shadow_ao_texture().unwrap(), w, h);
    let refl1 = read_rgba16f(&device, &queue, rt.reflection_texture().unwrap(), w, h);
    let gi1 = read_rgba16f(&device, &queue, rt.gi_texture().unwrap(), w, h);
    let idx_left = (16 * w + 7) as usize;
    let idx_right = (16 * w + 25) as usize;

    assert!(sh1[idx_left][0] < 0.05, "left floor must be shadowed in frame 1: {:?}", sh1[idx_left]);
    assert!(sh1[idx_right][0] > 0.95, "right floor must be lit in frame 1: {:?}", sh1[idx_right]);
    assert!(
        refl1[idx_left][0] > 1.5 && refl1[idx_left][0] > refl1[idx_left][2] * 3.0,
        "mirror reflection under red emissive roof must reflect bright red: {:?}",
        refl1[idx_left]
    );
    assert!(
        gi1[idx_left][0] > gi1[idx_left][2] * 2.0 && gi1[idx_left][0] > 0.4,
        "1-bounce diffuse GI under red roof must show strong red color bleed: {:?}",
        gi1[idx_left]
    );

    // Frame 2: dynamically move the TLAS instance to the RIGHT side + refit the BLAS in place!
    let refit_tris = [
        BlasTriangle::flat([-0.7, 0.0, -1.6], [0.7, 0.0, -1.6], [0.7, 0.0, 1.6]),
        BlasTriangle::flat([-0.7, 0.0, -1.6], [0.7, 0.0, 1.6], [-0.7, 0.0, 1.6]),
    ];
    roof_blas.refit(&refit_tris).unwrap();
    let mut right_inst = TlasInstance::new(0, translate(1.15, 1.5, 0.0));
    right_inst.albedo = [0.05, 0.9, 0.05];
    right_inst.emissive = [0.0, 3.0, 0.0];
    let scene2 = TlasScene::build(&[roof_blas], &[right_inst]).unwrap();
    rt.upload_scene(&device, &scene2).unwrap();

    let mut enc2 = device.create_command_encoder(&Default::default());
    rt.trace(&device, &mut enc2, &depth_view, (w, h), &params).unwrap();
    queue.submit(Some(enc2.finish()));

    let sh2 = read_rgba16f(&device, &queue, rt.shadow_ao_texture().unwrap(), w, h);
    let refl2 = read_rgba16f(&device, &queue, rt.reflection_texture().unwrap(), w, h);
    assert!(sh2[idx_left][0] > 0.95, "left floor must become lit after TLAS move: {:?}", sh2[idx_left]);
    assert!(sh2[idx_right][0] < 0.05, "right floor must become shadowed after TLAS move + refit: {:?}", sh2[idx_right]);
    assert!(
        refl2[idx_right][1] > 1.5 && refl2[idx_right][1] > refl2[idx_right][0] * 3.0,
        "reflection under moved green roof must now reflect bright green: {:?}",
        refl2[idx_right]
    );
}

#[test]
fn gpu_blas_pool_zero_reupload_and_hw_ray_query_parity() {
    let (device, queue) = crate::testing::device_with_optional().expect("GPU adapter required");
    let quad_tris = [
        BlasTriangle::flat([-0.6, 0.0, -1.5], [0.6, 0.0, -1.5], [0.6, 0.0, 1.5]),
        BlasTriangle::flat([-0.6, 0.0, -1.5], [0.6, 0.0, 1.5], [-0.6, 0.0, 1.5]),
    ];
    let translate = |x: f32, y: f32, z: f32| -> [f32; 16] {
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0]
    };
    let mut pool = GpuBlasPool::new(&device);
    let mut rt = TlasRayTracer::new(&device);

    let vp: [f32; 16] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0];
    let (w, h) = (32u32, 32u32);
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pool_rt_depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        depth_tex.as_image_copy(),
        bytemuck::cast_slice(&vec![0.5f32; (w * h) as usize]),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let depth_view = depth_tex.create_view(&Default::default());
    let params = TlasRayTraceParams {
        view_projection: vp,
        camera_pos: [0.0, 4.0, 0.0],
        depth_clear: 1.0,
        sun_direction: [0.0, 1.0, 0.0],
        sun_angular_radius: 0.0,
        sun_color: [2.0, 2.0, 2.0],
        sky_zenith: [0.05, 0.1, 0.3],
        sky_horizon: [0.05, 0.1, 0.25],
        surface_roughness: 0.0,
        ao_radius: 3.0,
        max_ray_distance: 100.0,
        shadow_samples: 1,
        ao_samples: 4,
        reflection_samples: 1,
        gi_samples: 0,
        gi_bounce_scale: 1.0,
        frame_seed: 1.0,
        normal_bias: 0.01,
        visibility_mask: u32::MAX,
    };

    // Animate across 3 positions using the SAME mesh ID (42): BLAS must upload ONCE, TLAS 3x!
    let mut last_sh = Vec::new();
    let mut last_refl = Vec::new();
    for &x in &[-1.15f32, 0.0, 1.15] {
        let mut enc = device.create_command_encoder(&Default::default());
        let bi = pool.ensure_blas(&device, &mut enc, 42, &quad_tris).unwrap().unwrap();
        let mut inst = TlasInstance::new(bi, translate(x, 1.5, 0.0));
        inst.albedo = [0.1, 0.9, 0.2];
        inst.emissive = [0.0, 3.0, 0.0];
        pool.update_tlas(&device, &mut enc, &[inst]).unwrap();
        let snap = pool.snapshot_scene();
        rt.trace_with_scene(&device, &mut enc, &snap, &depth_view, (w, h), &params).unwrap();
        queue.submit(Some(enc.finish()));
        last_sh = read_rgba16f(&device, &queue, rt.shadow_ao_texture().unwrap(), w, h);
        last_refl = read_rgba16f(&device, &queue, rt.reflection_texture().unwrap(), w, h);
    }
    assert_eq!(pool.blas_upload_count, 1, "static mesh BLAS must be built and uploaded only once across moves");
    assert_eq!(pool.tlas_update_count, 3, "each animated transform must perform a lightweight TLAS-only update");

    let idx_left = (16 * w + 7) as usize;
    let idx_right = (16 * w + 25) as usize;
    assert!(last_sh[idx_left][0] > 0.95, "left floor must be lit at final position: {:?}", last_sh[idx_left]);
    assert!(last_sh[idx_right][0] < 0.05, "right floor must be shadowed at final position: {:?}", last_sh[idx_right]);
    assert!(
        last_refl[idx_right][1] > 1.5 && last_refl[idx_right][1] > last_refl[idx_right][0] * 3.0,
        "persistent pool (hw_rt={}) must reflect moved green roof accurately: {:?}",
        pool.uses_hardware_ray_query(),
        last_refl[idx_right]
    );
}

#[test]
fn gpu_tlas_vndf_matte_cutoff_energy_conserving_occlude_and_hw_rt_skips_sw_tlas() {
    let (device, queue) = crate::testing::device_with_optional().expect("GPU adapter required");
    let half_floor = |x0: f32, x1: f32| {
        [
            BlasTriangle::flat([x0, 0.0, -2.5], [x0, 0.0, 2.5], [x1, 0.0, 2.5]),
            BlasTriangle::flat([x0, 0.0, -2.5], [x1, 0.0, 2.5], [x1, 0.0, -2.5]),
        ]
    };
    let roof = [
        BlasTriangle::flat([-2.5, 2.0, -2.5], [2.5, 2.0, -2.5], [2.5, 2.0, 2.5]),
        BlasTriangle::flat([-2.5, 2.0, -2.5], [2.5, 2.0, 2.5], [-2.5, 2.0, 2.5]),
    ];
    let id = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    let mut pool = GpuBlasPool::new(&device);
    let mut enc0 = device.create_command_encoder(&Default::default());
    let bi_left = pool.ensure_blas(&device, &mut enc0, 100, &half_floor(-2.5, 0.0)).unwrap().unwrap();
    let bi_right = pool.ensure_blas(&device, &mut enc0, 101, &half_floor(0.0, 2.5)).unwrap().unwrap();
    let bi_roof = pool.ensure_blas(&device, &mut enc0, 102, &roof).unwrap().unwrap();

    let mut left_matte = TlasInstance::new(bi_left, id);
    left_matte.albedo = [0.8, 0.8, 0.8];
    left_matte.roughness = 0.90; // > 0.75 matte cutoff!
    left_matte.metallic = 0.0;

    let mut right_metal = TlasInstance::new(bi_right, id);
    right_metal.albedo = [0.95, 0.95, 0.95];
    right_metal.roughness = 0.08; // glossy mirror!
    right_metal.metallic = 0.95;

    let mut roof_inst = TlasInstance::new(bi_roof, id);
    roof_inst.albedo = [0.9, 0.05, 0.05];
    roof_inst.emissive = [4.0, 0.05, 0.05];

    pool.update_tlas(&device, &mut enc0, &[left_matte, right_metal, roof_inst]).unwrap();
    let snap = pool.snapshot_scene();
    if pool.uses_hardware_ray_query() {
        assert_eq!(snap.tlas_node_count, 0, "HW ray-query path must skip CPU SW TLAS BVH build");
    }
    assert_eq!(snap.tlas_instance_count, 3);

    let vp: [f32; 16] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0];
    let (w, h) = (32u32, 32u32);
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vndf_depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        depth_tex.as_image_copy(),
        bytemuck::cast_slice(&vec![0.5f32; (w * h) as usize]),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let depth_v = depth_tex.create_view(&Default::default());
    let params = TlasRayTraceParams {
        view_projection: vp,
        camera_pos: [0.0, 4.0, 0.0],
        depth_clear: 1.0,
        sun_direction: [0.0, 1.0, 0.0],
        sun_angular_radius: 0.0,
        sun_color: [0.0, 0.0, 0.0],
        sky_zenith: [0.05, 0.1, 0.2],
        sky_horizon: [0.05, 0.1, 0.2],
        surface_roughness: 0.5,
        ao_radius: 2.0,
        max_ray_distance: 64.0,
        shadow_samples: 1,
        ao_samples: 2,
        reflection_samples: 4,
        gi_samples: 0,
        gi_bounce_scale: 1.0,
        frame_seed: 1.0,
        normal_bias: 0.01,
        visibility_mask: u32::MAX,
    };
    let mut rt = TlasRayTracer::new(&device);
    rt.trace_with_scene_gi(&device, &mut enc0, &snap, &depth_v, (w, h), &params, &[], None, true).unwrap();
    queue.submit(Some(enc0.finish()));

    let sao = read_rgba16f(&device, &queue, rt.shadow_ao_texture().unwrap(), w, h);
    let refl = read_rgba16f(&device, &queue, rt.reflection_texture().unwrap(), w, h);
    let idx_matte = (16 * w + 8) as usize;
    let idx_metal = (16 * w + 24) as usize;

    assert!(
        refl[idx_matte][0] < 0.01,
        "matte floor (rough=0.90 > 0.75) must skip reflection rays: {:?}",
        refl[idx_matte]
    );
    assert!(sao[idx_matte][2] < 0.01, "matte floor must report zero sky IBL occlusion: {:?}", sao[idx_matte]);
    assert!(
        (sao[idx_matte][3] - 0.90).abs() < 0.03,
        "primary lookup must export roughness 0.90 in sao.a: {:?}",
        sao[idx_matte]
    );

    assert!(
        refl[idx_metal][0] > 0.8,
        "glossy metal floor must reflect emissive red roof via Heitz VNDF: {:?}",
        refl[idx_metal]
    );
    assert!(
        sao[idx_metal][2] > 0.40,
        "glossy metal hit must export energy-conserving sky occlusion in sao.b: {:?}",
        sao[idx_metal]
    );
    assert!(
        (sao[idx_metal][3] - 0.08).abs() < 0.03,
        "primary lookup must export roughness 0.08 in sao.a: {:?}",
        sao[idx_metal]
    );
}
