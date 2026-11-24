struct MorphParams {
    vertex_count: u32,
    target_count: u32,
    weight_offset: u32,
    out_offset: u32,
};

@group(0) @binding(0) var<uniform> params: MorphParams;
@group(0) @binding(1) var<storage, read> base_verts: array<f32>;
@group(0) @binding(2) var<storage, read> deltas: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_verts: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let vi = gid.x;
    if (vi >= params.vertex_count) {
        return;
    }
    let src = vi * 10u;
    var px = base_verts[src + 0u];
    var py = base_verts[src + 1u];
    var pz = base_verts[src + 2u];
    var nx = base_verts[src + 3u];
    var ny = base_verts[src + 4u];
    var nz = base_verts[src + 5u];
    for (var t = 0u; t < params.target_count; t = t + 1u) {
        let w = weights[params.weight_offset + t];
        if (w != 0.0) {
            let d = (t * params.vertex_count + vi) * 6u;
            px = px + w * deltas[d + 0u];
            py = py + w * deltas[d + 1u];
            pz = pz + w * deltas[d + 2u];
            nx = nx + w * deltas[d + 3u];
            ny = ny + w * deltas[d + 4u];
            nz = nz + w * deltas[d + 5u];
        }
    }
    let nlen = sqrt(nx * nx + ny * ny + nz * nz);
    if (nlen > 1e-6) {
        let inv = 1.0 / nlen;
        nx = nx * inv;
        ny = ny * inv;
        nz = nz * inv;
    }
    let dst = (params.out_offset + vi) * 10u;
    out_verts[dst + 0u] = px;
    out_verts[dst + 1u] = py;
    out_verts[dst + 2u] = pz;
    out_verts[dst + 3u] = nx;
    out_verts[dst + 4u] = ny;
    out_verts[dst + 5u] = nz;
    out_verts[dst + 6u] = base_verts[src + 6u];
    out_verts[dst + 7u] = base_verts[src + 7u];
    out_verts[dst + 8u] = base_verts[src + 8u];
    out_verts[dst + 9u] = base_verts[src + 9u];
}
