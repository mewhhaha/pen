struct Tilemap {
    camera_vp: mat4x4f,
    origin_and_tile_size: vec4f,
    grid_size_and_atlas: vec4u,
    anim_time: vec4f,
    tint: vec4f,
}

@group(0) @binding(0) var<uniform> map: Tilemap;
@group(0) @binding(1) var<storage, read> cells: array<u32>;
@group(0) @binding(2) var atlas: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;

const QUAD = array<vec2f, 6>(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1),
                            vec2f(0, 1), vec2f(1, 0), vec2f(1, 1));

struct Vertex {
    @builtin(position) position: vec4f,
    @location(0) grid_position: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> Vertex {
    let grid = QUAD[vertex] * vec2f(map.grid_size_and_atlas.xy);
    let world = map.origin_and_tile_size.xy + grid * map.origin_and_tile_size.zw * vec2f(1, -1);
    return Vertex(map.camera_vp * vec4f(world, 0, 1), grid);
}

@fragment
fn fs_main(vertex: Vertex) -> @location(0) vec4f {
    let grid = max(vertex.grid_position, vec2f(0));
    let cell = min(vec2u(grid), map.grid_size_and_atlas.xy - 1u);
    let bits = cells[cell.y * map.grid_size_and_atlas.x + cell.x];
    let id = bits & 0xffffu;
    if id == 0u {
        return vec4f(0);
    }
    let frames = max((bits >> 19u) & 31u, 1u);
    let frame = floor(map.anim_time.x * map.anim_time.y);
    let frame_count = f32(frames);
    let animation_index = u32(frame - frame_count * floor(frame / frame_count));
    let index = id - 1u + animation_index;
    let atlas_grid = map.grid_size_and_atlas.zw;
    if index / atlas_grid.x >= atlas_grid.y {
        return vec4f(0);
    }

    var local = clamp(grid - vec2f(cell), vec2f(0), vec2f(1));
    if (bits & (1u << 18u)) != 0u {
        local = vec2f(local.y, 1.0 - local.x);
    }
    let flip = (vec2u(bits) & vec2u(1u << 16u, 1u << 17u)) != vec2u(0);
    local = select(local, 1.0 - local, flip);

    let texture_size = vec2f(textureDimensions(atlas));
    let tile_size = texture_size / vec2f(atlas_grid);
    let tile_origin = vec2f(vec2u(index % atlas_grid.x, index / atlas_grid.x)) * tile_size;
    let texel = tile_origin + clamp(local * tile_size, vec2f(0.5), tile_size - 0.5);
    return textureSampleLevel(atlas, atlas_sampler, texel / texture_size, 0.0) * map.tint;
}
