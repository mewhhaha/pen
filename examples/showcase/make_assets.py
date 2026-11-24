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
