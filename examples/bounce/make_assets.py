"""Writes the example's meshes and sound into src/assets: a box and a ball,
each face counterclockwise seen from outside (checked), and a short blip."""
import json
import math
import struct
import wave


def numbers(values):
    rounded = [round(value, 5) for value in values]
    return [int(value) if float(value).is_integer() else value for value in rounded]


def check_winding(positions, normals, indices):
    for at in range(0, len(indices), 3):
        a, b, c = [positions[3 * index:3 * index + 3] for index in indices[at:at + 3]]
        one = [b[k] - a[k] for k in range(3)]
        two = [c[k] - a[k] for k in range(3)]
        cross = (one[1] * two[2] - one[2] * two[1], one[2] * two[0] - one[0] * two[2], one[0] * two[1] - one[1] * two[0])
        normal = [sum(normals[3 * index + k] for index in indices[at:at + 3]) for k in range(3)]
        if sum(value * value for value in cross) > 1e-12:
            assert sum(cross[k] * normal[k] for k in range(3)) > 0, f"triangle {at // 3} faces inward"


def write_mesh(path, positions, normals, uvs, indices):
    check_winding(positions, normals, indices)
    mesh = {"positions": numbers(positions), "normals": numbers(normals), "uvs": numbers(uvs), "indices": indices}
    with open(path, "w") as file:
        file.write(json.dumps(mesh) + "\n")


def box():
    """From -1 to 1 on each axis, four corners a face."""
    faces = [((1, 0, 0), (0, 1, 0)), ((-1, 0, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, -1)),
             ((0, -1, 0), (0, 0, 1)), ((0, 0, 1), (0, 1, 0)), ((0, 0, -1), (0, 1, 0))]
    positions, normals, uvs, indices = [], [], [], []
    for normal, up in faces:
        side = (up[1] * normal[2] - up[2] * normal[1], up[2] * normal[0] - up[0] * normal[2], up[0] * normal[1] - up[1] * normal[0])
        first = len(positions) // 3
        for across, along in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            positions += [normal[k] + across * side[k] + along * up[k] for k in range(3)]
            normals += list(normal)
            uvs += [(across + 1) / 2, (1 - along) / 2]
        indices += [first, first + 1, first + 2, first, first + 2, first + 3]
    write_mesh("src/assets/box.mesh.json", positions, normals, uvs, indices)


def ball(around=24, rings=12):
    """Radius 1, `around` segments round it and `rings` from pole to pole."""
    positions, normals, uvs, indices = [], [], [], []
    for ring in range(rings + 1):
        down = math.pi * ring / rings
        for step in range(around + 1):
            turn = 2 * math.pi * step / around
            point = (math.sin(down) * math.cos(turn), math.cos(down), -math.sin(down) * math.sin(turn))
            positions += point
            normals += point
            uvs += [step / around, ring / rings]
    for ring in range(rings):
        for step in range(around):
            a = ring * (around + 1) + step
            b, c = a + 1, a + around + 1
            if ring != 0:
                indices += [a, c, b]
            if ring != rings - 1:
                indices += [b, c, c + 1]
    write_mesh("src/assets/ball.mesh.json", positions, normals, uvs, indices)


def blip(rate=48000, seconds=0.07, pitch=660.0):
    """A tone falling away fast, 16-bit mono."""
    samples = int(rate * seconds)
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * pitch * at / rate) * math.exp(-at / (rate * 0.018))))
        for at in range(samples)
    )
    with wave.open("src/assets/bounce.wav", "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(frames)


box()
ball()
blip()
