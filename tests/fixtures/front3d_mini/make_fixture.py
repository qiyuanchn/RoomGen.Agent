"""Generate the ``front3d_mini`` fixture: a strict-format 3D-FRONT scene.

The fixture is *hand-authored* but written out by this script so the geometry
stays readable and the committed ``scene.json`` stays diffable::

    python tests/fixtures/front3d_mini/make_fixture.py          # rewrite in place
    python tests/fixtures/front3d_mini/make_fixture.py --print  # summary + expected poses

What it produces (all coordinates in the dataset's own frame: **Y-up, metres**,
floor plane ``y = 0``):

* two rooms -- ``LivingDiningRoom-mini1`` (x 0..3.5) and ``BedRoom-mini2``
  (x 3.5..7.0), both z 0..3.2, ceiling 2.6 m
* shell meshes with the real dataset's key set (``uid`` / ``aid`` / ``jid`` /
  ``xyz`` / ``normal`` / ``uv`` / ``faces`` / ``material`` / ``type`` /
  ``constructid``): floors, ceilings, outer walls with a window opening, the
  partition wall with a door gap, a baseboard strip and the window glass
* materials with colours and one with a texture reference (so the
  ``3D-FRONT-textures/<hash>/texture.png`` lookup is exercised)
* furniture: two sofas and a bed with the dataset's ``pos`` / ``rot`` / ``scale``,
  one item flagged ``valid: false``, one placed outside every room, and one whose
  model folder is deliberately absent (mirrors the partially public
  ``3D-FUTURE-model`` set)
* three 3D-FUTURE models under ``3D-FUTURE-model/<jid>/``: ``raw_model.obj``
  (Y-up, base-centre origin, ``y in [0, height]``), ``model.mtl``, ``texture.png``

The expected world poses (repo/Blender frame, ``(x, -z, y)``) are printed with
``--print`` and asserted in ``tests/test_front3d_math.py`` /
``tests/test_front3d_replay.sh``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(HERE, "3D-FUTURE-model")
TEXTURE_ROOT = os.path.join(HERE, "3D-FRONT-textures")

SOFA_JID = "0f1b6f2e-1c1e-4a5a-9b1f-0000000000a1"
BED_JID = "0f1b6f2e-1c1e-4a5a-9b1f-0000000000b2"
NIGHTSTAND_JID = "0f1b6f2e-1c1e-4a5a-9b1f-0000000000c3"
MISSING_JID = "0f1b6f2e-1c1e-4a5a-9b1f-ffffffffffff"
WALL_TEXTURE_HASH = "9c1f2ab40d5e77c1a3b8d2e5f6071829"

ROOM_A = "LivingDiningRoom-mini1"
ROOM_B = "BedRoom-mini2"
WALL_HEIGHT = 2.6
DOOR_Z0, DOOR_Z1 = 1.15, 2.05
# windows in room A's west wall (x = 0) and room B's east wall (x = 7).  The
# east one sits away from the camera's +X depth probe (see
# tests/check_front3d_build.py: from camera 0001 the +X ray goes along front
# z = 1.6, and this pane spans z 2.4..3.0, i.e. 25-39 deg off that ray).
WINDOW_Z0, WINDOW_Z1 = 1.20, 2.20
WINDOW_Y0, WINDOW_Y1 = 0.90, 1.90
WINDOW_B_Z0, WINDOW_B_Z1 = 2.40, 3.00


# --------------------------------------------------------------------------- #
# tiny 3D helpers (rectangles -> triangles with normals and UVs)
# --------------------------------------------------------------------------- #
def rectangle(corners, uv_scale=(1.0, 1.0)):
    """One planar quad -> (xyz flat, normal flat, uv flat, faces flat).

    ``corners`` are four ``(x, y, z)`` points in order (counter-clockwise seen
    from the normal side).  The datatset stores one normal and one UV per vertex
    and triangles as flat index triples.
    """
    a, b, c, d = corners
    u = [b[i] - a[i] for i in range(3)]
    v = [d[i] - a[i] for i in range(3)]
    normal = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
    length = math.sqrt(sum(component * component for component in normal)) or 1.0
    normal = [component / length for component in normal]
    xyz = []
    for point in corners:
        xyz.extend(round(float(value), 6) for value in point)
    uvs = [
        0.0, 0.0,
        round(math.dist(a, b) * uv_scale[0], 6), 0.0,
        round(math.dist(a, b) * uv_scale[0], 6), round(math.dist(a, d) * uv_scale[1], 6),
        0.0, round(math.dist(a, d) * uv_scale[1], 6),
    ]
    return xyz, [round(value, 6) for value in normal] * 4, uvs, [0, 1, 2, 0, 2, 3]


def box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box -> one 6-quad mesh in the dataset's flat format."""
    xyz, faces = [], []
    normals, uvs = [], []
    base = 0
    quads = [
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],      # +y
        [(x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0)],      # -y
        [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],      # -z
        [(x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1)],      # +z
        [(x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (x0, y0, z0)],      # -x
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],      # +x
    ]
    for corners in quads:
        quad_xyz, quad_normal, quad_uv, quad_faces = rectangle(corners)
        xyz.extend(quad_xyz)
        normals.extend(quad_normal)
        uvs.extend(quad_uv)
        faces.extend(index + base for index in quad_faces)
        base += 4
    return xyz, normals, uvs, faces


# --------------------------------------------------------------------------- #
# PNG writer (no dependency on PIL, so the fixture regenerates anywhere)
# --------------------------------------------------------------------------- #
def write_png(path, width, height, pixel_fn):
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixel_fn(x, y))

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    data = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b""))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def checker(color_a, color_b, size=16, cell=4):
    def pixel(x, y):
        return color_a if ((x // cell) + (y // cell)) % 2 == 0 else color_b
    return lambda: (size, size, pixel)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def write_obj(path, name, parts, material_name):
    """Write a 3D-FUTURE-style ``raw_model.obj`` (Y-up, base-centre origin)."""
    lines = [f"# 3D-FUTURE mini fixture model: {name}",
             f"mtllib model.mtl", f"o {name}"]
    vertices, uvs, normals, faces = [], [], [], []
    for xyz, face_normals, face_uvs, indices in parts:
        vertex_base = len(vertices) + 1
        uv_base = len(uvs) + 1
        for i in range(0, len(xyz), 3):
            vertices.append((xyz[i], xyz[i + 1], xyz[i + 2]))
        for i in range(0, len(face_uvs), 2):
            uvs.append((face_uvs[i], face_uvs[i + 1]))
        for i in range(0, len(face_normals), 3):
            normals.append((face_normals[i], face_normals[i + 1], face_normals[i + 2]))
        for i in range(0, len(indices), 3):
            a, b, c = (indices[i] + vertex_base, indices[i + 1] + vertex_base, indices[i + 2] + vertex_base)
            au, bu, cu = (indices[i] + uv_base, indices[i + 1] + uv_base, indices[i + 2] + uv_base)
            an, bn, cn = (indices[i] + 1, indices[i + 1] + 1, indices[i + 2] + 1)
            faces.append((f"{a}/{au}/{an}", f"{b}/{bu}/{bn}", f"{c}/{cu}/{cn}"))
    for x, y, z in vertices:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for u, v in uvs:
        lines.append(f"vt {u:.6f} {v:.6f}")
    for x, y, z in normals:
        lines.append(f"vn {x:.6f} {y:.6f} {z:.6f}")
    lines.append(f"usemtl {material_name}")
    lines.append("s off")
    for a, b, c in faces:
        lines.append(f"f {a} {b} {c}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(vertices), len(faces)


def write_mtl(path, material_name, color):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f"# 3D-FUTURE mini fixture material\n"
            f"newmtl {material_name}\n"
            f"Kd {color[0]:.4f} {color[1]:.4f} {color[2]:.4f}\n"
            f"Ka 0.1000 0.1000 0.1000\n"
            f"Tf 1.0000 1.0000 1.0000\n"
            f"map_Kd texture.png\n"
        )


def build_models():
    """The three 3D-FUTURE models, with base-centre origins (y from 0)."""
    made = {}
    # sofa: seat box + back box, 1.80 x 0.75 x 0.84, origin at the base centre
    sofa_parts = [
        box(-0.90, 0.90, 0.00, 0.42, -0.42, 0.42),
        box(-0.90, 0.90, 0.42, 0.75, -0.42, -0.24),
        box(-0.90, -0.72, 0.00, 0.62, -0.42, 0.42),
        box(0.72, 0.90, 0.00, 0.62, -0.42, 0.42),
    ]
    made["sofa"] = (SOFA_JID, "sofa", sofa_parts, (0.72, 0.30, 0.24))
    # bed: frame + mattress + headboard, 1.60 x 0.60 x 2.00, origin at the base
    bed_parts = [
        box(-0.80, 0.80, 0.00, 0.30, -1.00, 1.00),
        box(-0.75, 0.75, 0.30, 0.60, -0.95, 0.85),
        box(-0.80, 0.80, 0.00, 1.10, 0.85, 1.00),
    ]
    made["bed"] = (BED_JID, "bed", bed_parts, (0.55, 0.60, 0.75))
    # nightstand: 0.45 x 0.50 x 0.45 box with a drawer front
    nightstand_parts = [
        box(-0.225, 0.225, 0.00, 0.50, -0.225, 0.225),
        box(-0.19, 0.19, 0.06, 0.20, 0.225, 0.245),
    ]
    made["nightstand"] = (NIGHTSTAND_JID, "nightstand", nightstand_parts, (0.85, 0.82, 0.78))

    report = {}
    for key, (jid, name, parts, color) in made.items():
        folder = os.path.join(MODEL_ROOT, jid)
        vertices, faces = write_obj(os.path.join(folder, "raw_model.obj"), name, parts, name + "_material")
        write_mtl(os.path.join(folder, "model.mtl"), name + "_material", color)
        size, _, pixel = checker(
            tuple(int(round(c * 255)) for c in color),
            tuple(int(round(min(1.0, c + 0.18) * 255)) for c in color),
        )()
        write_png(os.path.join(folder, "texture.png"), size, size, pixel)
        report[key] = {"jid": jid, "vertices": vertices, "faces": faces}
    return report, made


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
def build_scene(models):
    mesh_entries = []
    material_entries = []

    def add_material(uid, color, texture=None, jid=None):
        entry = {
            "uid": uid, "aid": [], "jid": jid or "", "texture": texture or "",
            "normaltexture": "", "color": list(color) + [255], "seamWidth": 0,
            "useColor": True,
            "normalUVTransform": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "contentType": ["material", "paint"],
        }
        material_entries.append(entry)
        return uid

    mat_floor = add_material("front3d-mini/floor", [176, 140, 96])
    mat_wall = add_material(
        "front3d-mini/wall", [232, 234, 233],
        texture=f"https://3d-front.example/texture/{WALL_TEXTURE_HASH}/texture.png",
    )
    mat_ceiling = add_material("front3d-mini/ceiling", [248, 248, 246])
    mat_trim = add_material("front3d-mini/trim", [245, 245, 242])
    mat_glass = add_material("front3d-mini/glass", [198, 226, 240])

    counter = {"n": 0}

    # every shell mesh: (constructid, dataset type, quad corners, material).
    # The west wall is split around the window opening and the partition around
    # the door gap, exactly as the dataset represents openings in its planes.
    mesh_plan = [
        ("300001", "Floor", [(0.0, 0.0, 0.0), (3.5, 0.0, 0.0), (3.5, 0.0, 3.2), (0.0, 0.0, 3.2)], mat_floor),
        ("300002", "Floor", [(3.5, 0.0, 0.0), (7.0, 0.0, 0.0), (7.0, 0.0, 3.2), (3.5, 0.0, 3.2)], mat_floor),
        ("300003", "Ceiling", [(0.0, WALL_HEIGHT, 3.2), (3.5, WALL_HEIGHT, 3.2),
                               (3.5, WALL_HEIGHT, 0.0), (0.0, WALL_HEIGHT, 0.0)], mat_ceiling),
        ("300004", "Ceiling", [(3.5, WALL_HEIGHT, 3.2), (7.0, WALL_HEIGHT, 3.2),
                               (7.0, WALL_HEIGHT, 0.0), (3.5, WALL_HEIGHT, 0.0)], mat_ceiling),
        ("300101", "WallOuter", [(0.0, 0.0, 0.0), (7.0, 0.0, 0.0), (7.0, WALL_HEIGHT, 0.0),
                                 (0.0, WALL_HEIGHT, 0.0)], mat_wall),
        ("300102", "WallOuter", [(7.0, 0.0, 3.2), (0.0, 0.0, 3.2), (0.0, WALL_HEIGHT, 3.2),
                                 (7.0, WALL_HEIGHT, 3.2)], mat_wall),
        ("300111", "WallOuter", [(0.0, 0.0, 0.0), (0.0, 0.0, WINDOW_Z0), (0.0, WALL_HEIGHT, WINDOW_Z0),
                                 (0.0, WALL_HEIGHT, 0.0)], mat_wall),
        ("300112", "WallOuter", [(0.0, 0.0, WINDOW_Z1), (0.0, 0.0, 3.2), (0.0, WALL_HEIGHT, 3.2),
                                 (0.0, WALL_HEIGHT, WINDOW_Z1)], mat_wall),
        ("300113", "WallOuter", [(0.0, 0.0, WINDOW_Z0), (0.0, 0.0, WINDOW_Z1),
                                 (0.0, WINDOW_Y0, WINDOW_Z1), (0.0, WINDOW_Y0, WINDOW_Z0)], mat_wall),
        ("300114", "WallOuter", [(0.0, WINDOW_Y1, WINDOW_Z0), (0.0, WINDOW_Y1, WINDOW_Z1),
                                 (0.0, WALL_HEIGHT, WINDOW_Z1), (0.0, WALL_HEIGHT, WINDOW_Z0)], mat_wall),
        ("300121", "WallOuter", [(0.0, 0.0, 3.2), (0.0, 0.0, 0.0), (0.0, WALL_HEIGHT, 0.0),
                                 (0.0, WALL_HEIGHT, 3.2)], mat_wall),
        ("300131", "WallOuter", [(7.0, 0.0, 0.0), (7.0, 0.0, WINDOW_B_Z0), (7.0, WALL_HEIGHT, WINDOW_B_Z0),
                                 (7.0, WALL_HEIGHT, 0.0)], mat_wall),
        ("300132", "WallOuter", [(7.0, 0.0, WINDOW_B_Z1), (7.0, 0.0, 3.2), (7.0, WALL_HEIGHT, 3.2),
                                 (7.0, WALL_HEIGHT, WINDOW_B_Z1)], mat_wall),
        ("300133", "WallOuter", [(7.0, 0.0, WINDOW_B_Z0), (7.0, 0.0, WINDOW_B_Z1),
                                 (7.0, WINDOW_Y0, WINDOW_B_Z1), (7.0, WINDOW_Y0, WINDOW_B_Z0)], mat_wall),
        ("300134", "WallOuter", [(7.0, WINDOW_Y1, WINDOW_B_Z0), (7.0, WINDOW_Y1, WINDOW_B_Z1),
                                 (7.0, WALL_HEIGHT, WINDOW_B_Z1), (7.0, WALL_HEIGHT, WINDOW_B_Z0)], mat_wall),
        ("300201", "WallInner", [(3.5, 0.0, 0.0), (3.5, 0.0, DOOR_Z0), (3.5, WALL_HEIGHT, DOOR_Z0),
                                 (3.5, WALL_HEIGHT, 0.0)], mat_wall),
        ("300202", "WallInner", [(3.5, 0.0, DOOR_Z1), (3.5, 0.0, 3.2), (3.5, WALL_HEIGHT, 3.2),
                                 (3.5, WALL_HEIGHT, DOOR_Z1)], mat_wall),
        ("300301", "Baseboard", [(0.0, 0.0, 0.0), (3.5, 0.0, 0.0), (3.5, 0.10, 0.0),
                                 (0.0, 0.10, 0.0)], mat_trim),
        ("300401", "Window", [(0.0, WINDOW_Y0, WINDOW_Z0), (0.0, WINDOW_Y0, WINDOW_Z1),
                              (0.0, WINDOW_Y1, WINDOW_Z1), (0.0, WINDOW_Y1, WINDOW_Z0)], mat_glass),
        ("300402", "Window", [(7.0, WINDOW_Y0, WINDOW_B_Z1), (7.0, WINDOW_Y0, WINDOW_B_Z0),
                              (7.0, WINDOW_Y1, WINDOW_B_Z0), (7.0, WINDOW_Y1, WINDOW_B_Z1)], mat_glass),
    ]

    mesh_uids = {}
    for constructid, kind, corners, material in mesh_plan:
        xyz, normals, uvs, faces = rectangle(corners)
        uid = f"{constructid}/{counter['n']}"
        counter["n"] += 1
        mesh_entries.append({
            "aid": [], "jid": "", "uid": uid,
            "xyz": xyz, "normal": normals, "uv": uvs, "faces": faces,
            "material": material, "type": kind, "constructid": constructid,
        })
        mesh_uids[constructid] = uid

    def refs(constructids):
        """Mesh children carry the mesh's ``uid`` (``<constructid>/<n>``)."""
        return [
            {"ref": mesh_uids[constructid], "pos": [0, 0, 0], "rot": [0, 0, 0, 1], "scale": [1, 1, 1]}
            for constructid in constructids
        ]

    # ---- furniture ------------------------------------------------------- #
    yaw90 = [0.0, round(math.sin(math.pi / 4), 12), 0.0, round(math.cos(math.pi / 4), 12)]
    furniture = [
        {"uid": "100/model", "jid": SOFA_JID, "aid": [], "title": "sofa/double seat sofa",
         "type": "standard", "category": "Sofa", "valid": True,
         "size": [1.8, 0.75, 0.84], "bbox": [[1.8, 0.75, 0.84]],
         "sourceCategoryId": "e7a0801e-7abd-4634-b3bc-8777342d124e"},
        {"uid": "101/model", "jid": SOFA_JID, "aid": [], "title": "sofa/single seat sofa",
         "type": "standard", "category": "Sofa", "valid": True,
         "size": [1.8, 0.75, 0.84], "bbox": [[1.8, 0.75, 0.84]]},
        {"uid": "131/model", "jid": BED_JID, "aid": [], "title": "bed/king-size bed",
         "type": "standard", "valid": True, "size": [1.6, 1.1, 2.0],
         "bbox": [[1.6, 1.1, 2.0]]},
        {"uid": "132/model", "jid": NIGHTSTAND_JID, "aid": [], "title": "table/night table",
         "type": "standard", "category": "Table", "valid": True,
         "size": [0.45, 0.5, 0.45], "bbox": [[0.45, 0.5, 0.45]]},
        {"uid": "140/model", "jid": MISSING_JID, "aid": [], "title": "lighting/pendant light",
         "type": "standard", "category": "Lighting", "valid": True,
         "size": [0.5, 0.6, 0.5], "bbox": [[0.5, 0.6, 0.5]]},
        {"uid": "150/model", "jid": SOFA_JID, "aid": [], "title": "sofa/ottoman",
         "type": "standard", "category": "Sofa", "valid": False,
         "size": [0.6, 0.45, 0.6], "bbox": [[0.6, 0.45, 0.6]]},
        {"uid": "160/model", "jid": NIGHTSTAND_JID, "aid": [], "title": "table/side table",
         "type": "standard", "category": "Table", "valid": True,
         "size": [0.45, 0.5, 0.45], "bbox": [[0.45, 0.5, 0.45]]},
    ]

    def child(ref, pos, rot=(0.0, 0.0, 0.0, 1.0), scale=(1.0, 1.0, 1.0)):
        return {"ref": ref, "pos": [round(float(v), 6) for v in pos],
                "rot": [float(v) for v in rot],
                "scale": [round(float(v), 6) for v in scale]}

    rooms = [
        {
            "type": "LivingDiningRoom", "instanceid": ROOM_A, "size": 11.2,
            "pos": [0, 0, 0], "rot": [0, 0, 0, 1], "scale": [1, 1, 1], "empty": 0,
            "children": (
                refs(["300001", "300003", "300101", "300102", "300111", "300112",
                      "300113", "300114", "300121", "300201", "300301", "300401"])
                + [child("100/model", (1.2, 0.0, 1.6)),
                   child("101/model", (1.2, 0.0, 0.75), scale=(0.6, 1.0, 1.0)),
                   child("140/model", (1.75, 2.15, 1.6)),
                   child("150/model", (2.6, 0.0, 2.4))]
            ),
        },
        {
            "type": "BedRoom", "instanceid": ROOM_B, "size": 11.2,
            "pos": [0, 0, 0], "rot": [0, 0, 0, 1], "scale": [1, 1, 1], "empty": 0,
            "children": (
                refs(["300002", "300004", "300131", "300132", "300133", "300134", "300202", "300402"])
                + [child("131/model", (5.4, 0.0, 1.7), rot=yaw90, scale=(1.0, 0.8, 1.1)),
                   child("132/model", (4.35, 0.0, 2.9), rot=yaw90),
                   # outside every room: the bridge must drop it, not emit an
                   # invalid spec (real data places doors/windows like this)
                   child("160/model", (20.0, 0.0, 1.6))]
            ),
        },
    ]

    return {
        "uid": "front3d-mini-0001",
        "jobid": "front3d-mini",
        "design_version": "1",
        "code_version": "1",
        "north_vector": [0.0, 0.0, -1.0],
        "furniture": furniture,
        "mesh": mesh_entries,
        "material": material_entries,
        "lights": [
            {"uid": "140/light", "nodeType": "VRayLight", "roomId": ROOM_B,
             "src_position": [5.25, 2.2, 1.6], "direction": [0, -1, 0], "up_vector": [0, 0, 1],
             "color_temperature": 4000, "multiplier": 80000, "enabled": True, "type": 0},
        ],
        "extension": {
            "door": [{"type": "interiorDoor", "roomId": ROOM_A, "dir": "outerLeft", "ref": ["100/model"]}],
            "snapshots": [
                {"pos": [1.75, 1.5, 2.4], "target": [1.75, 1.5, 0.0], "up": [0, 1, 0],
                 "near": 0.1, "far": 100, "fov": 60, "name": "living", "thumbnail": ""},
                {"pos": [5.6, 1.5, 2.6], "target": [5.6, 1.5, 0.0], "up": [0, 1, 0],
                 "near": 0.1, "far": 100, "fov": 60, "name": "bedroom", "thumbnail": ""},
            ],
        },
        "scene": {"room": rooms},
        "version": "2.0",
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def expected_report():
    """The hand-computed facts the tests assert (repo frame, metres)."""
    half = math.sin(math.pi / 4)
    yaw90 = [0.0, half, 0.0, half]
    sofa_pos, bed_pos = (1.2, 0.0, 1.6), (5.4, 0.0, 1.7)
    bed_scale = (1.0, 0.8, 1.1)
    # corner of the bed model's bounding box under the dataset transform:
    # scale -> yaw 90 about +Y -> translate, then front (x, y, z) -> repo (x, -z, y)
    local = (0.8, 1.1, 1.0)
    scaled = tuple(local[i] * bed_scale[i] for i in range(3))
    rotated = (scaled[2], scaled[1], -scaled[0])
    front = tuple(rotated[i] + bed_pos[i] for i in range(3))
    return {
        "yaw90_quaternion": yaw90,
        "sofa": {"front_pos": list(sofa_pos), "repo_pos": [sofa_pos[0], -sofa_pos[2], sofa_pos[1]],
                 "world_aabb": [[0.3, -2.02, 0.0], [2.1, -1.18, 0.75]]},
        "bed": {"front_pos": list(bed_pos),
                "repo_pos": [bed_pos[0], -bed_pos[2], bed_pos[1]],
                "corner_local": list(local),
                "corner_front": list(front),
                "corner_repo": [front[0], -front[2], front[1]],
                "world_aabb": [[4.3, -2.5, 0.0], [6.5, -0.9, 0.88]]},
        "cameras": {"0000": [1.75, -1.6, 1.5], "0001": [5.25, -1.6, 1.5]},
        "wall_height": WALL_HEIGHT,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print", action="store_true", help="print the expected poses and exit")
    args = parser.parse_args(argv)
    if args.print:
        print(json.dumps(expected_report(), indent=2))
        return 0

    model_report, _ = build_models()
    scene = build_scene(model_report)
    scene_path = os.path.join(HERE, "scene.json")
    with open(scene_path, "w", encoding="utf-8") as handle:
        json.dump(scene, handle, indent=1)
        handle.write("\n")

    texture_dir = os.path.join(TEXTURE_ROOT, WALL_TEXTURE_HASH)
    size, _, pixel = checker((214, 206, 194), (188, 178, 164), size=16, cell=8)()
    write_png(os.path.join(texture_dir, "texture.png"), size, size, pixel)

    print(f"wrote {scene_path}")
    print(f"  {len(scene['mesh'])} meshes, {len(scene['material'])} materials, "
          f"{len(scene['furniture'])} furniture entries, {len(scene['scene']['room'])} rooms")
    for key, info in sorted(model_report.items()):
        print(f"  model {key}: {info['jid']} ({info['vertices']} verts, {info['faces']} faces)")
    print(f"wrote {os.path.join(TEXTURE_ROOT, WALL_TEXTURE_HASH, 'texture.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
