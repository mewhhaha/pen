"""Generates deterministic procedural skinned hero rig, meshlet-culled sentinel bot,
arena geometry, visual shader graph, UI font/shader, materials, and SFX for Citadel Striker."""
import json
import math
import os
import shutil
import struct
import wave
import zlib

ASSETS = "src/assets"
os.makedirs(ASSETS, exist_ok=True)


def numbers(values):
    rounded = [round(v, 5) for v in values]
    return [int(v) if float(v).is_integer() else v for v in rounded]


def write_png(path, w, h, pixel_fn):
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            r, g, b, a = pixel_fn(x, y, w, h)
            raw.extend((r, g, b, a))

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)


def append_box(pos, norm, uv, idx, joints, weights, center, half, joint_id=None):
    faces = [
        ((1, 0, 0), (0, 1, 0)),
        ((-1, 0, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 0, -1)),
        ((0, -1, 0), (0, 0, 1)),
        ((0, 0, 1), (0, 1, 0)),
        ((0, 0, -1), (0, 1, 0)),
    ]
    cx, cy, cz = center
    hx, hy, hz = half
    for normal, up in faces:
        side = (
            up[1] * normal[2] - up[2] * normal[1],
            up[2] * normal[0] - up[0] * normal[2],
            up[0] * normal[1] - up[1] * normal[0],
        )
        base = len(pos) // 3
        for ax, ay in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            px = cx + (normal[0] + ax * side[0] + ay * up[0]) * hx
            py = cy + (normal[1] + ax * side[1] + ay * up[1]) * hy
            pz = cz + (normal[2] + ax * side[2] + ay * up[2]) * hz
            pos += [px, py, pz]
            norm += list(normal)
            uv += [(ax + 1) / 2, (1 - ay) / 2]
            if joint_id is not None:
                joints += [joint_id, 0, 0, 0]
                weights += [1.0, 0.0, 0.0, 0.0]
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]


def inv_translate(tx, ty, tz):
    return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        -tx, -ty, -tz, 1,
    ]


def make_skinned_hero(path):
    """Articulated 6-bone hero rig (pelvis, torso, head, left_leg, right_leg, sword_arm).
    World rest positions:
      0 pelvis:    (0, 0.90, 0)
      1 torso:     (0, 1.35, 0)  (local +0.45 y from pelvis)
      2 head:      (0, 1.85, 0)  (local +0.50 y from torso)
      3 left_leg:  (-0.20, 0.45, 0) (local (-0.20, -0.45, 0) from pelvis)
      4 right_leg: ( 0.20, 0.45, 0) (local ( 0.20, -0.45, 0) from pelvis)
      5 sword_arm: ( 0.44, 1.45, 0) (local ( 0.44,  0.10, 0) from torso)
    """
    pos, norm, uv, idx, joints, weights = [], [], [], [], [], []
    # Pelvis block
    append_box(pos, norm, uv, idx, joints, weights, (0.0, 0.90, 0.0), (0.28, 0.16, 0.18), 0)
    # Armored torso + pauldrons
    append_box(pos, norm, uv, idx, joints, weights, (0.0, 1.35, 0.0), (0.34, 0.30, 0.22), 1)
    # Visored helm
    append_box(pos, norm, uv, idx, joints, weights, (0.0, 1.85, 0.0), (0.20, 0.20, 0.22), 2)
    # Left leg
    append_box(pos, norm, uv, idx, joints, weights, (-0.18, 0.45, 0.0), (0.12, 0.42, 0.14), 3)
    # Right leg
    append_box(pos, norm, uv, idx, joints, weights, (0.18, 0.45, 0.0), (0.12, 0.42, 0.14), 4)
    # Right sword arm
    append_box(pos, norm, uv, idx, joints, weights, (0.46, 1.28, -0.06), (0.11, 0.30, 0.12), 5)

    skeleton = [
        {"name": "pelvis", "translation": [0.0, 0.90, 0.0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
        {"name": "torso", "parent": 0, "translation": [0.0, 0.45, 0.0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
        {"name": "head", "parent": 1, "translation": [0.0, 0.50, 0.0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
        {"name": "left_leg", "parent": 0, "translation": [-0.18, -0.45, 0.0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
        {"name": "right_leg", "parent": 0, "translation": [0.18, -0.45, 0.0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
        {"name": "sword_arm", "parent": 1, "translation": [0.46, 0.10, 0.0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
    ]
    skin = [
        {"node": 0, "inverseBind": inv_translate(0.0, 0.90, 0.0)},
        {"node": 1, "inverseBind": inv_translate(0.0, 1.35, 0.0)},
        {"node": 2, "inverseBind": inv_translate(0.0, 1.85, 0.0)},
        {"node": 3, "inverseBind": inv_translate(-0.18, 0.45, 0.0)},
        {"node": 4, "inverseBind": inv_translate(0.18, 0.45, 0.0)},
        {"node": 5, "inverseBind": inv_translate(0.46, 1.45, 0.0)},
    ]
    mesh = {
        "positions": numbers(pos),
        "normals": numbers(norm),
        "uvs": numbers(uv),
        "uvs1": numbers(uv),
        "indices": idx,
        "joints": joints,
        "weights": numbers(weights),
        "skeleton": skeleton,
        "skin": skin,
        "clips": [],
    }
    with open(path, "w") as f:
        f.write(json.dumps(mesh) + "\n")


def make_box(path):
    pos, norm, uv, idx = [], [], [], []
    append_box(pos, norm, uv, idx, None, None, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), None)
    mesh = {"positions": numbers(pos), "normals": numbers(norm), "uvs": numbers(uv), "uvs1": numbers(uv), "indices": idx}
    with open(path, "w") as f:
        f.write(json.dumps(mesh) + "\n")


def make_blade(path):
    pos, norm, uv, idx = [], [], [], []
    # Hilt + tapered glowing blade pointing along -Z
    append_box(pos, norm, uv, idx, None, None, (0.0, 0.0, -0.70), (0.06, 0.03, 0.72), None)
    append_box(pos, norm, uv, idx, None, None, (0.0, 0.0, -0.05), (0.20, 0.05, 0.05), None)
    mesh = {"positions": numbers(pos), "normals": numbers(norm), "uvs": numbers(uv), "uvs1": numbers(uv), "indices": idx}
    with open(path, "w") as f:
        f.write(json.dumps(mesh) + "\n")


def make_sphere(path, around=20, rings=12):
    """Dense UV sphere (> 400 triangles) so Pen's GPU meshlet cluster culling activates."""
    pos, norm, uv, idx = [], [], [], []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(around + 1):
            theta = 2 * math.pi * s / around
            pt = (
                math.sin(phi) * math.cos(theta),
                math.cos(phi),
                -math.sin(phi) * math.sin(theta),
            )
            pos += pt
            norm += pt
            uv += [s / around, r / rings]
    for r in range(rings):
        for s in range(around):
            a = r * (around + 1) + s
            b, c = a + 1, a + around + 1
            if r != 0:
                idx += [a, c, b]
            if r != rings - 1:
                idx += [b, c, c + 1]
    mesh = {"positions": numbers(pos), "normals": numbers(norm), "uvs": numbers(uv), "uvs1": numbers(uv), "indices": idx}
    with open(path, "w") as f:
        f.write(json.dumps(mesh) + "\n")


def make_wav(path, seconds, sample_fn, rate=48000):
    n = int(rate * seconds)
    frames = b"".join(
        struct.pack("<h", max(-32767, min(32767, int(32767 * sample_fn(i / rate, seconds)))))
        for i in range(n)
    )
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


# Meshes
make_box(f"{ASSETS}/box.mesh.json")
make_skinned_hero(f"{ASSETS}/hero.mesh.json")
make_sphere(f"{ASSETS}/sentinel.mesh.json", around=20, rings=12)
make_blade(f"{ASSETS}/blade.mesh.json")

# Stock full-featured PBR shader (supports vertex_skinned, LTC area lights, cookies, decals)
shutil.copyfile("../../crates/pen/templates/new/src/assets/lit.wgsl", f"{ASSETS}/lit.wgsl")

# Procedural textures
write_png(
    f"{ASSETS}/stone.png",
    32,
    32,
    lambda x, y, w, h: (150, 156, 168, 255)
    if ((x // 8) + (y // 8)) % 2 == 0
    else (122, 128, 140, 255),
)
write_png(
    f"{ASSETS}/portal.png",
    32,
    32,
    lambda x, y, w, h: (
        int(60 + 160 * (x / w)),
        int(180 + 70 * (y / h)),
        255,
        255,
    ),
)
write_png(
    f"{ASSETS}/rune_albedo.png",
    32,
    32,
    lambda x, y, w, h: (
        (255, 196, 70, 230)
        if 100 <= (x - 15.5) ** 2 + (y - 15.5) ** 2 <= 175
        else (0, 0, 0, 0)
    ),
)
write_png(f"{ASSETS}/rune_normal.png", 16, 16, lambda x, y, w, h: (128, 128, 255, 255))
write_png(f"{ASSETS}/rune_orm.png", 16, 16, lambda x, y, w, h: (255, 60, 220, 255))
write_png(
    f"{ASSETS}/flare.png",
    16,
    16,
    lambda x, y, w, h: (
        255,
        240,
        180,
        max(0, min(255, int(255 * max(0.0, 1.0 - math.hypot(x - 7.5, y - 7.5) / 7.5)))),
    ),
)

# Crisp 5x7 ASCII 32..95 bitmap font atlas (8x8 cells = 64x64)
FONT_5X7 = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    "!": (4, 4, 4, 4, 4, 0, 4),
    "+": (0, 4, 4, 31, 4, 4, 0),
    "-": (0, 0, 0, 31, 0, 0, 0),
    ".": (0, 0, 0, 0, 0, 12, 12),
    "/": (1, 1, 2, 4, 8, 16, 16),
    ":": (0, 12, 12, 0, 12, 12, 0),
    "0": (14, 17, 19, 21, 25, 17, 14),
    "1": (4, 12, 4, 4, 4, 4, 14),
    "2": (14, 17, 1, 2, 4, 8, 31),
    "3": (30, 1, 1, 14, 1, 1, 30),
    "4": (2, 6, 10, 18, 31, 2, 2),
    "5": (31, 16, 30, 1, 1, 17, 14),
    "6": (14, 16, 30, 17, 17, 17, 14),
    "7": (31, 1, 2, 4, 8, 8, 8),
    "8": (14, 17, 17, 14, 17, 17, 14),
    "9": (14, 17, 17, 15, 1, 2, 12),
    "A": (14, 17, 17, 31, 17, 17, 17),
    "B": (30, 17, 17, 30, 17, 17, 30),
    "C": (14, 17, 16, 16, 16, 17, 14),
    "D": (28, 18, 17, 17, 17, 18, 28),
    "E": (31, 16, 16, 30, 16, 16, 31),
    "F": (31, 16, 16, 30, 16, 16, 16),
    "G": (14, 17, 16, 23, 17, 17, 15),
    "H": (17, 17, 17, 31, 17, 17, 17),
    "I": (14, 4, 4, 4, 4, 4, 14),
    "J": (7, 2, 2, 2, 18, 18, 12),
    "K": (17, 18, 20, 24, 20, 18, 17),
    "L": (16, 16, 16, 16, 16, 16, 31),
    "M": (17, 27, 21, 21, 17, 17, 17),
    "N": (17, 25, 21, 19, 17, 17, 17),
    "O": (14, 17, 17, 17, 17, 17, 14),
    "P": (30, 17, 17, 30, 16, 16, 16),
    "Q": (14, 17, 17, 17, 21, 18, 13),
    "R": (30, 17, 17, 30, 20, 18, 17),
    "S": (15, 16, 16, 14, 1, 1, 30),
    "T": (31, 4, 4, 4, 4, 4, 4),
    "U": (17, 17, 17, 17, 17, 17, 14),
    "V": (17, 17, 17, 17, 17, 10, 4),
    "W": (17, 17, 17, 21, 21, 27, 17),
    "X": (17, 17, 10, 4, 10, 17, 17),
    "Y": (17, 17, 10, 4, 4, 4, 4),
    "Z": (31, 1, 2, 4, 8, 16, 31),
}


def font_pixel(x, y, w, h):
    code = 32 + (y // 8) * 8 + (x // 8)
    ch = chr(code) if 32 <= code < 96 else " "
    lx, ly = x % 8, y % 8
    rows = FONT_5X7.get(ch, (0, 0, 0, 0, 0, 0, 0))
    if 1 <= lx <= 5 and 0 <= ly < 7 and ((rows[ly] >> (5 - lx)) & 1):
        return (255, 255, 255, 255)
    return (0, 0, 0, 0)


write_png(f"{ASSETS}/font.png", 64, 64, font_pixel)

UI_WGSL = """@group(0) @binding(0) var<uniform> screen: vec4f;
struct SpriteData {
  rect: vec4f,
  uv: vec4f,
  color: vec4f,
  parameters: vec4f,
};
@group(0) @binding(1) var<storage, read> sprites: array<SpriteData>;
@group(0) @binding(2) var atlas: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;

struct Out {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) local_uv: vec2f,
  @location(2) color: vec4f,
  @location(3) params: vec4f,
  @location(4) size: vec2f,
};

@vertex fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
  let corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
  );
  let s = sprites[i];
  let c = corners[v];
  let px = s.rect.xy + c * s.rect.zw;
  let ndc = vec2f(px.x / max(screen.x, 1.0) * 2.0 - 1.0, 1.0 - px.y / max(screen.y, 1.0) * 2.0);
  let uv = mix(s.uv.xy, s.uv.zw, c);
  return Out(vec4f(ndc, 0.0, 1.0), uv, c, s.color, s.parameters, s.rect.zw);
}

@fragment fn fragment(in: Out) -> @location(0) vec4f {
  if (in.params.x > 0.5) {
    let sample = textureSample(atlas, atlas_sampler, in.uv);
    let alpha = step(0.35, sample.a) * in.color.a;
    if (alpha <= 0.01) {
      discard;
    }
    return vec4f(in.color.rgb * sample.rgb, alpha);
  }
  let half_size = in.size * 0.5;
  let p = (in.local_uv - vec2f(0.5)) * in.size;
  let radius = min(in.params.y, min(half_size.x, half_size.y));
  let q = abs(p) - (half_size - vec2f(radius));
  let dist = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - radius;
  let edge = clamp(0.5 - dist, 0.0, 1.0);
  if (edge <= 0.01) {
    discard;
  }
  let border = smoothstep(-2.0, -0.5, dist) * in.params.z;
  let rgb = mix(in.color.rgb, min(in.color.rgb * 1.45 + vec3f(0.18), vec3f(1.0)), border);
  return vec4f(rgb, in.color.a * edge);
}
"""
with open(f"{ASSETS}/ui.wgsl", "w") as f:
    f.write(UI_WGSL)

# Visual ShaderGraph (.matgraph.json) for arcane crystal shards & citadel core
portal_graph = {
    "name": "citadel_portal",
    "nodes": [
        {"id": "uv", "op": "uv", "ui_pos": [0.0, 0.0]},
        {"id": "tex", "op": "texture2_d", "ui_pos": [120.0, 0.0], "uv": {"node": "uv"}, "fallback_color": [0.25, 0.85, 1.0]},
        {"id": "rgb", "op": "swizzle", "ui_pos": [240.0, 0.0], "input": {"node": "tex"}, "mask": "xyz"},
        {"id": "t", "op": "time", "ui_pos": [120.0, 120.0]},
        {"id": "s", "op": "sin", "ui_pos": [220.0, 120.0], "x": {"node": "t"}},
        {"id": "pulse", "op": "clamp", "ui_pos": [320.0, 120.0], "x": {"node": "s"}, "min": 0.25, "max": 0.95},
        {"id": "rim", "op": "fresnel", "ui_pos": [320.0, 180.0], "power": 3.0},
        {"id": "boost", "op": "add", "ui_pos": [420.0, 140.0], "a": {"node": "pulse"}, "b": {"node": "rim"}},
        {"id": "glow", "op": "mul", "ui_pos": [500.0, 120.0], "a": {"node": "rgb"}, "b": {"node": "boost"}},
        {"id": "rough", "op": "float_const", "ui_pos": [320.0, 240.0], "value": 0.14},
        {"id": "coat", "op": "float_const", "ui_pos": [320.0, 300.0], "value": 0.9},
        {"id": "irid", "op": "float_const", "ui_pos": [320.0, 360.0], "value": 0.85},
        {
            "id": "out",
            "op": "standard_surface_output",
            "ui_pos": [620.0, 80.0],
            "base_color": {"node": "rgb"},
            "roughness": {"node": "rough"},
            "emissive": {"node": "glow"},
            "clearcoat": {"node": "coat"},
            "iridescence": {"node": "irid"},
        },
    ],
}
with open(f"{ASSETS}/portal.matgraph.json", "w") as f:
    json.dump(portal_graph, f, indent=2)
    f.write("\n")

materials = {
    "stone.material.json": {
        "shader": "lit.wgsl",
        "texture": "stone.png",
        "color": [0.68, 0.70, 0.76, 1.0],
        "lighting": "lit",
        "decals": True,
    },
    "pillar.material.json": {
        "shader": "lit.wgsl",
        "texture": "stone.png",
        "color": [0.46, 0.50, 0.58, 1.0],
        "lighting": "lit",
        "decals": True,
        "surface": {"clearcoat": 0.25, "clearcoatRoughness": 0.35},
    },
    "hero.material.json": {
        "shader": "lit.wgsl",
        "color": [0.24, 0.56, 0.92, 1.0],
        "lighting": "lit",
        "surface": {"clearcoat": 0.6, "clearcoatRoughness": 0.2},
    },
    "sentinel.material.json": {
        "shader": "lit.wgsl",
        "color": [0.85, 0.20, 0.22, 1.0],
        "lighting": "lit",
        "surface": {"clearcoat": 0.7, "clearcoatRoughness": 0.18, "emissiveStrength": 0.6},
    },
    "sentinel_hit.material.json": {
        "shader": "lit.wgsl",
        "color": [1.0, 0.92, 0.65, 1.0],
        "lighting": "unlit",
        "surface": {"emissiveStrength": 3.5},
    },
    "blade.material.json": {
        "shader": "lit.wgsl",
        "color": [0.35, 0.95, 1.0, 1.0],
        "lighting": "unlit",
        "surface": {"emissiveStrength": 4.2},
    },
    "portal.material.json": {
        "shader": "portal.matgraph.json",
        "texture": "portal.png",
        "color": [0.85, 0.95, 1.0, 1.0],
        "lighting": "lit",
        "surface": {
            "clearcoat": 0.9,
            "clearcoatRoughness": 0.1,
            "iridescence": 0.85,
            "iridescenceIor": 1.65,
            "iridescenceThicknessMin": 220.0,
            "iridescenceThicknessMax": 540.0,
        },
    },
    "flare.material.json": {
        "shader": "lit.wgsl",
        "texture": "flare.png",
        "color": [1.0, 0.85, 0.35, 1.0],
        "lighting": "unlit",
        "blend": True,
    },
    "rune.material.json": {
        "shader": "lit.wgsl",
        "texture": "rune_albedo.png",
        "normalTexture": "rune_normal.png",
        "ormTexture": "rune_orm.png",
        "color": [1.0, 1.0, 1.0, 1.0],
        "lighting": "lit",
    },
    "ui_panel.material.json": {
        "shader": "ui.wgsl",
        "color": [1.0, 1.0, 1.0, 1.0],
        "lighting": "unlit",
        "blend": True,
    },
    "ui_font.material.json": {
        "shader": "ui.wgsl",
        "texture": "font.png",
        "color": [1.0, 1.0, 1.0, 1.0],
        "lighting": "unlit",
        "blend": True,
    },
}

for name, body in materials.items():
    with open(f"{ASSETS}/{name}", "w") as f:
        json.dump(body, f, indent=2)
        f.write("\n")

# Procedural SFX clips
make_wav(f"{ASSETS}/step.wav", 0.06, lambda t, d: 0.25 * math.sin(2 * math.pi * (140 - 80 * t / d) * t) * math.exp(-t / 0.018))
make_wav(f"{ASSETS}/swing.wav", 0.12, lambda t, d: 0.35 * math.sin(2 * math.pi * (520 - 300 * t / d) * t) * math.exp(-t / 0.04))
make_wav(f"{ASSETS}/hit.wav", 0.14, lambda t, d: 0.45 * (math.sin(2 * math.pi * 240 * t) + 0.5 * math.sin(2 * math.pi * 610 * t)) * math.exp(-t / 0.035))
make_wav(f"{ASSETS}/pickup.wav", 0.16, lambda t, d: 0.35 * math.sin(2 * math.pi * (660 + 440 * (t / d)) * t) * math.exp(-t / 0.06))

catalog = {
    "meshes": ["box.mesh.json", "hero.mesh.json", "sentinel.mesh.json", "blade.mesh.json"],
    "materials": list(materials.keys()),
    "sounds": ["step.wav", "swing.wav", "hit.wav", "pickup.wav"],
}
with open(f"{ASSETS}/catalog.json", "w") as f:
    json.dump(catalog, f, indent=2)
    f.write("\n")
print("Generated citadel assets cleanly.")
