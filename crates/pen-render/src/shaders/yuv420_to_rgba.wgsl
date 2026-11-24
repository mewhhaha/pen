// GPU planar YUV420p -> RGBA8 converter supporting BT.709 and BT.601 in both
// limited (studio TV) and full (PC) ranges, with configurable 4:2:0 chroma
// siting (Centered / JPEG, Left / MPEG-2, TopLeft / PAL-DV) and hardware
// bilinear chroma upsampling.

struct YuvParams {
    // 0 = BT.709 limited, 1 = BT.709 full, 2 = BT.601 limited, 3 = BT.601 full
    colorspace: u32,
    // 0 = Centered (JPEG), 1 = Left (MPEG-2 / BT.709), 2 = TopLeft (PAL-DV)
    siting: u32,
    // Chroma UV shift in normalized texture coordinates: (du, dv).
    chroma_shift_u: f32,
    chroma_shift_v: f32,
}

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var u_tex: texture_2d<f32>;
@group(0) @binding(2) var v_tex: texture_2d<f32>;
@group(0) @binding(3) var yuv_sampler: sampler;
@group(0) @binding(4) var<uniform> params: YuvParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
    let y = 1.0 - f32(idx & 2u) * 2.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv_c = clamp(
        in.uv + vec2<f32>(params.chroma_shift_u, params.chroma_shift_v),
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
    let y_raw = textureSampleLevel(y_tex, yuv_sampler, in.uv, 0.0).r;
    let u_raw = textureSampleLevel(u_tex, yuv_sampler, uv_c, 0.0).r;
    let v_raw = textureSampleLevel(v_tex, yuv_sampler, uv_c, 0.0).r;

    let limited = (params.colorspace == 0u) || (params.colorspace == 2u);
    let bt601 = (params.colorspace == 2u) || (params.colorspace == 3u);

    var y = y_raw;
    var cb = u_raw - (128.0 / 255.0);
    var cr = v_raw - (128.0 / 255.0);
    if limited {
        y = clamp((y_raw - (16.0 / 255.0)) * (255.0 / 219.0), 0.0, 1.0);
        cb = cb * (255.0 / 224.0);
        cr = cr * (255.0 / 224.0);
    }

    var rgb: vec3<f32>;
    if bt601 {
        // ITU-R BT.601 (Kr = 0.299, Kb = 0.114)
        rgb = vec3<f32>(
            y + 1.402 * cr,
            y - 0.344136 * cb - 0.714136 * cr,
            y + 1.772 * cb,
        );
    } else {
        // ITU-R BT.709 (Kr = 0.2126, Kb = 0.0722)
        rgb = vec3<f32>(
            y + 1.5748 * cr,
            y - 0.187324 * cb - 0.468124 * cr,
            y + 1.8556 * cb,
        );
    }
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
