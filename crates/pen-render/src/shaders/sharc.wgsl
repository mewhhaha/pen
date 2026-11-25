// Consumers supply sharc_u: SharcUniform (uniform), sharc_keys and sharc_accum
// (read_write storage arrays of atomic<u32>), and sharc_resolved (read storage
// array of vec4f). A query-only entry point needs no accumulation binding.
struct SharcUniform {
    camera_and_voxel: vec4f,
    params: vec4f,
    counts: vec4u,
    pad: vec4u,
}

struct SharcKey {
    home: u32,
    tag: u32,
}

const SHARC_PROBE_STEPS: u32 = 8u;
const SHARC_MAX_SAMPLES: u32 = 256u;

fn sharc_level(pos: vec3f, u: SharcUniform) -> u32 {
    let dist = length(pos - u.camera_and_voxel.xyz);
    let near = max(u.camera_and_voxel.w * 8.0, 1e-3);
    return u32(clamp(floor(log2(max(dist / near, 1.0))), 0.0, 12.0));
}

fn sharc_voxel_size(level: u32, u: SharcUniform) -> f32 {
    return u.camera_and_voxel.w * exp2(f32(level));
}

fn sharc_normal_face(nrm: vec3f) -> u32 {
    let a = abs(nrm);
    if (a.x >= a.y && a.x >= a.z) {
        return select(0u, 1u, nrm.x < 0.0);
    }
    if (a.y >= a.z) {
        return select(2u, 3u, nrm.y < 0.0);
    }
    return select(4u, 5u, nrm.z < 0.0);
}

fn sharc_hash_mix(value: u32) -> u32 {
    var h = value;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn sharc_hash_cell(cell: vec3u, level: u32, face: u32, seed: u32) -> u32 {
    let x = sharc_hash_mix(seed ^ cell.x);
    let y = sharc_hash_mix(x ^ cell.y);
    let z = sharc_hash_mix(y ^ cell.z);
    return sharc_hash_mix(z ^ (level * 6u + face));
}

fn sharc_hash_key(pos: vec3f, nrm: vec3f, u: SharcUniform) -> SharcKey {
    let level = sharc_level(pos, u);
    let safe_cell = clamp(floor(pos / sharc_voxel_size(level, u)), vec3f(-1e8), vec3f(1e8));
    let cell = bitcast<vec3u>(vec3i(safe_cell));
    let face = sharc_normal_face(nrm);
    let home = sharc_hash_cell(cell, level, face, 0x9e3779b9u) & (u.counts.x - 1u);
    let fingerprint = sharc_hash_cell(cell, level, face, 0x85ebca6bu);
    // One CAS publishes the complete tag. Zero marks empty, and the low bits
    // hold the exact face so opposite normals cannot match, even on a hash
    // collision. The default home contributes 17 bits and the fingerprint
    // contributes 29 bits within each face. Probing compares tags at up to
    // eight slots, so distinct cells on the same face can still alias.
    return SharcKey(home, (fingerprint & ~7u) | (face + 1u));
}

// __SHARC_MUTABLE__
// RGB is resolved radiance. Alpha is age plus one, with zero for a cache miss.
// Consumers can query and deposit in the same dispatch: resolved storage is
// read-only until the later resolve pass.
fn sharc_query(pos: vec3f, nrm: vec3f) -> vec4f {
    if (sharc_u.counts.z == 0u) {
        return vec4f(0.0);
    }
    let key = sharc_hash_key(pos, nrm, sharc_u);
    let mask = sharc_u.counts.x - 1u;
    for (var probe = 0u; probe < min(SHARC_PROBE_STEPS, sharc_u.counts.x); probe += 1u) {
        let slot = (key.home + probe) & mask;
        if (atomicLoad(&sharc_keys[slot]) == key.tag) {
            return sharc_resolved[slot];
        }
    }
    return vec4f(0.0);
}

fn sharc_finite_nonnegative(v: f32, limit: f32) -> f32 {
    return select(clamp(v, 0.0, limit), 0.0, (bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u);
}

fn sharc_find_or_claim(key: SharcKey) -> u32 {
    let capacity = sharc_u.counts.x;
    let mask = capacity - 1u;
    for (var attempt = 0u; attempt <= SHARC_PROBE_STEPS; attempt += 1u) {
        var empty = capacity;
        // Eviction makes holes in probe chains. Find an existing tag before
        // trying to claim the first empty slot.
        for (var probe = 0u; probe < min(SHARC_PROBE_STEPS, capacity); probe += 1u) {
            let slot = (key.home + probe) & mask;
            let tag = atomicLoad(&sharc_keys[slot]);
            if (tag == key.tag) {
                return slot;
            }
            if (tag == 0u && empty == capacity) {
                empty = slot;
            }
        }
        if (empty == capacity) {
            return capacity;
        }
        let claim = atomicCompareExchangeWeak(&sharc_keys[empty], 0u, key.tag);
        if (claim.exchanged || claim.old_value == key.tag) {
            return empty;
        }
        // Retry after a spurious failure or a competing claim. During a
        // consumer pass, a slot can only transition from empty to occupied.
    }
    return capacity;
}

fn sharc_accumulate(pos: vec3f, nrm: vec3f, radiance: vec3f) {
    if (sharc_u.counts.z == 0u) {
        return;
    }
    let limit = sharc_u.params.z;
    let safe_rad = vec3f(
        sharc_finite_nonnegative(radiance.x, limit),
        sharc_finite_nonnegative(radiance.y, limit),
        sharc_finite_nonnegative(radiance.z, limit),
    );
    let slot = sharc_find_or_claim(sharc_hash_key(pos, nrm, sharc_u));
    if (slot == sharc_u.counts.x) {
        return;
    }

    let base = slot * 4u;
    var count = atomicLoad(&sharc_accum[base + 3u]);
    // Reserve a sample before touching the sums. At most 256 accepted samples
    // reach each bucket even when many invocations deposit concurrently.
    loop {
        if (count >= SHARC_MAX_SAMPLES) {
            return;
        }
        let ticket = atomicCompareExchangeWeak(&sharc_accum[base + 3u], count, count + 1u);
        if (ticket.exchanged) {
            break;
        }
        count = ticket.old_value;
    }
    let fixed = vec3u(safe_rad * sharc_u.params.y);
    atomicAdd(&sharc_accum[base], fixed.x);
    atomicAdd(&sharc_accum[base + 1u], fixed.y);
    atomicAdd(&sharc_accum[base + 2u], fixed.z);
}

// __SHARC_ENTRYPOINTS__
@group(0) @binding(0) var<uniform> sharc_u: SharcUniform;
@group(0) @binding(1) var<storage, read_write> sharc_keys: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> sharc_accum: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sharc_resolved: array<vec4f>;

@compute @workgroup_size(64)
fn sharc_resolve(@builtin(global_invocation_id) id: vec3u) {
    let slot = id.x;
    if (slot >= sharc_u.counts.x || sharc_u.counts.z == 0u) {
        return;
    }
    let base = slot * 4u;
    let r = atomicExchange(&sharc_accum[base], 0u);
    let g = atomicExchange(&sharc_accum[base + 1u], 0u);
    let b = atomicExchange(&sharc_accum[base + 2u], 0u);
    let count = atomicExchange(&sharc_accum[base + 3u], 0u);
    let prev = sharc_resolved[slot];
    if (count > 0u) {
        let avg = vec3f(f32(r), f32(g), f32(b)) / (sharc_u.params.y * f32(count));
        let filtered = select(avg, mix(prev.rgb, avg, sharc_u.params.x), prev.a > 0.5);
        sharc_resolved[slot] = vec4f(filtered, 1.0);
    } else if (prev.a > 0.5) {
        let age = prev.a + 1.0;
        if (age > f32(sharc_u.counts.y)) {
            sharc_resolved[slot] = vec4f(0.0);
            atomicStore(&sharc_keys[slot], 0u);
        } else {
            sharc_resolved[slot] = vec4f(prev.rgb, age);
        }
    } else if (atomicLoad(&sharc_keys[slot]) != 0u) {
        atomicStore(&sharc_keys[slot], 0u);
    }
}

@compute @workgroup_size(64)
fn sharc_clear(@builtin(global_invocation_id) id: vec3u) {
    let slot = id.x;
    if (slot >= sharc_u.counts.x) {
        return;
    }
    let base = slot * 4u;
    atomicStore(&sharc_keys[slot], 0u);
    atomicStore(&sharc_accum[base], 0u);
    atomicStore(&sharc_accum[base + 1u], 0u);
    atomicStore(&sharc_accum[base + 2u], 0u);
    atomicStore(&sharc_accum[base + 3u], 0u);
    sharc_resolved[slot] = vec4f(0.0);
}
