"""Generates deterministic procedural meshes, textures, visual shader graph, materials,
and audio chime for the Ink + Pen High-Fidelity Showcase & Scene Editor."""
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


def write_mesh(path, positions, normals, uvs, indices):
    mesh = {
        "positions": numbers(positions),
        "normals": numbers(normals),
        "uvs": numbers(uvs),
        "uvs1": numbers(uvs),
        "indices": indices,
    }
    with open(path, "w") as f:
        f.write(json.dumps(mesh) + "\n")


def make_box(path):
    faces = [
        ((1, 0, 0), (0, 1, 0)),
        ((-1, 0, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 0, -1)),
        ((0, -1, 0), (0, 0, 1)),
        ((0, 0, 1), (0, 1, 0)),
        ((0, 0, -1), (0, 1, 0)),
    ]
    pos, norm, uv, idx = [], [], [], []
    for normal, up in faces:
        side = (
            up[1] * normal[2] - up[2] * normal[1],
            up[2] * normal[0] - up[0] * normal[2],
            up[0] * normal[1] - up[1] * normal[0],
        )
        base = len(pos) // 3
        for ax, ay in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            pos += [normal[k] + ax * side[k] + ay * up[k] for k in range(3)]
            norm += list(normal)
            uv += [(ax + 1) / 2, (1 - ay) / 2]
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    write_mesh(path, pos, norm, uv, idx)


def make_sphere(path, around=24, rings=12):
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
    write_mesh(path, pos, norm, uv, idx)


def make_chime(path, rate=48000, seconds=0.09, freq=880.0):
    n = int(rate * seconds)
    frames = b"".join(
        struct.pack(
            "<h",
            int(10000 * math.sin(2 * math.pi * freq * i / rate) * math.exp(-i / (rate * 0.025))),
        )
        for i in range(n)
    )
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


# Meshes
make_box(f"{ASSETS}/box.mesh.json")
make_sphere(f"{ASSETS}/sphere.mesh.json")

# Stock full-featured PBR shader (LTC area lights, cookies, lightmaps, iridescence)
shutil.copyfile("../../crates/pen/templates/new/src/assets/lit.wgsl", f"{ASSETS}/lit.wgsl")

# Procedural textures
write_png(
    f"{ASSETS}/checker.png",
    32,
    32,
    lambda x, y, w, h: (215, 220, 226, 255)
    if ((x // 8) + (y // 8)) % 2 == 0
    else (165, 174, 186, 255),
)
write_png(
    f"{ASSETS}/crystal.png",
    32,
    32,
    lambda x, y, w, h: (
        int(90 + 110 * (x / w)),
        int(180 + 60 * (y / h)),
        255,
        255,
    ),
)
write_png(
    f"{ASSETS}/cookie.png",
    32,
    32,
    lambda x, y, w, h: (
        (255, 220, 140, 255)
        if ((x - 15.5) ** 2 + (y - 15.5) ** 2) < 120
        and abs(x - 15.5) > 2.2
        and abs(y - 15.5) > 2.2
        else (20, 28, 45, 255)
    ),
)
write_png(
    f"{ASSETS}/decal_albedo.png",
    32,
    32,
    lambda x, y, w, h: (
        (60, 215, 235, 230)
        if 85 <= (x - 15.5) ** 2 + (y - 15.5) ** 2 <= 165
        else (0, 0, 0, 0)
    ),
)
write_png(f"{ASSETS}/decal_normal.png", 16, 16, lambda x, y, w, h: (128, 128, 255, 255))
write_png(f"{ASSETS}/decal_orm.png", 16, 16, lambda x, y, w, h: (255, 70, 210, 255))

# Procedural crisp 5x7 ASCII 32..95 bitmap font atlas (8x8 grid of 8x8 cells = 64x64)
FONT_5X7 = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    "!": (4, 4, 4, 4, 4, 0, 4),
    "+": (0, 4, 4, 31, 4, 4, 0),
    "-": (0, 0, 0, 31, 0, 0, 0),
    ".": (0, 0, 0, 0, 0, 12, 12),
    "/": (1, 1, 2, 4, 8, 16, 16),
    ":": (0, 12, 12, 0, 12, 12, 0),
    "[": (14, 8, 8, 8, 8, 8, 14),
    "]": (14, 2, 2, 2, 2, 2, 14),
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
    col = x // 8
    row = y // 8
    code = 32 + row * 8 + col
    ch = chr(code) if 32 <= code < 96 else " "
    lx = x % 8
    ly = y % 8
    rows = FONT_5X7.get(ch, (0, 0, 0, 0, 0, 0, 0))
    if 1 <= lx <= 5 and 0 <= ly < 7:
        if (rows[ly] >> (5 - lx)) & 1:
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

# Visual ShaderGraph (.matgraph.json) for the iridescent crystal
crystal_graph = {
    "name": "showcase_crystal",
    "nodes": [
        {"id": "uv", "op": "uv", "ui_pos": [0.0, 0.0]},
        {
            "id": "tex",
            "op": "texture2_d",
            "ui_pos": [120.0, 0.0],
            "uv": {"node": "uv"},
            "fallback_color": [0.35, 0.8, 1.0],
        },
        {"id": "rgb", "op": "swizzle", "ui_pos": [240.0, 0.0], "input": {"node": "tex"}, "mask": "xyz"},
        {"id": "t", "op": "time", "ui_pos": [120.0, 120.0]},
        {"id": "s", "op": "sin", "ui_pos": [220.0, 120.0], "x": {"node": "t"}},
        {"id": "pulse", "op": "clamp", "ui_pos": [320.0, 120.0], "x": {"node": "s"}, "min": 0.15, "max": 0.85},
        {"id": "rim", "op": "fresnel", "ui_pos": [320.0, 180.0], "power": 3.5},
        {"id": "boost", "op": "add", "ui_pos": [420.0, 140.0], "a": {"node": "pulse"}, "b": {"node": "rim"}},
        {"id": "glow", "op": "mul", "ui_pos": [500.0, 120.0], "a": {"node": "rgb"}, "b": {"node": "boost"}},
        {"id": "rough", "op": "float_const", "ui_pos": [320.0, 240.0], "value": 0.16},
        {"id": "coat", "op": "float_const", "ui_pos": [320.0, 300.0], "value": 0.85},
        {"id": "irid", "op": "float_const", "ui_pos": [320.0, 360.0], "value": 0.9},
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
with open(f"{ASSETS}/crystal.matgraph.json", "w") as f:
    json.dump(crystal_graph, f, indent=2)
    f.write("\n")

materials = {
    "floor.material.json": {
        "shader": "lit.wgsl",
        "texture": "checker.png",
        "color": [0.78, 0.8, 0.84, 1.0],
        "lighting": "lit",
        "decals": True,
    },
    "monolith.material.json": {
        "shader": "lit.wgsl",
        "color": [0.42, 0.45, 0.52, 1.0],
        "lighting": "lit",
        "decals": True,
        "surface": {"clearcoat": 0.35, "clearcoatRoughness": 0.25},
    },
    "crystal.material.json": {
        "shader": "crystal.matgraph.json",
        "texture": "crystal.png",
        "color": [0.85, 0.95, 1.0, 1.0],
        "lighting": "lit",
        "surface": {
            "clearcoat": 0.85,
            "clearcoatRoughness": 0.12,
            "iridescence": 0.9,
            "iridescenceIor": 1.75,
            "iridescenceThicknessMin": 200.0,
            "iridescenceThicknessMax": 520.0,
        },
    },
    "beacon.material.json": {
        "shader": "lit.wgsl",
        "color": [1.0, 0.38, 0.15, 1.0],
        "lighting": "unlit",
        "surface": {"emissiveStrength": 5.0},
    },
    "foliage.material.json": {
        "shader": "lit.wgsl",
        "color": [0.22, 0.62, 0.3, 1.0],
        "lighting": "lit",
    },
    "glass_red.material.json": {
        "shader": "lit.wgsl",
        "color": [0.95, 0.14, 0.12, 0.45],
        "lighting": "lit",
        "blend": True,
    },
    "glass_blue.material.json": {
        "shader": "lit.wgsl",
        "color": [0.12, 0.36, 0.98, 0.55],
        "lighting": "lit",
        "blend": True,
    },
    "gizmo_x.material.json": {"shader": "lit.wgsl", "color": [0.98, 0.22, 0.2, 1.0], "lighting": "unlit", "overlay": True},
    "gizmo_y.material.json": {"shader": "lit.wgsl", "color": [0.22, 0.95, 0.32, 1.0], "lighting": "unlit", "overlay": True},
    "gizmo_z.material.json": {"shader": "lit.wgsl", "color": [0.24, 0.52, 1.0, 1.0], "lighting": "unlit", "overlay": True},
    "gizmo_sel.material.json": {"shader": "lit.wgsl", "color": [1.0, 0.88, 0.2, 1.0], "lighting": "unlit", "overlay": True},
    "cookie.material.json": {"shader": "lit.wgsl", "texture": "cookie.png", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit"},
    "decal.material.json": {
        "shader": "lit.wgsl",
        "texture": "decal_albedo.png",
        "normalTexture": "decal_normal.png",
        "ormTexture": "decal_orm.png",
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

make_chime(f"{ASSETS}/chime.wav")

catalog = {
    "meshes": ["box.mesh.json", "sphere.mesh.json"],
    "materials": list(materials.keys()),
    "sounds": ["chime.wav"],
}
with open(f"{ASSETS}/catalog.json", "w") as f:
    json.dump(catalog, f, indent=2)
    f.write("\n")
print("Generated showcase assets cleanly.")
