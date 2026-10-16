"""Builtin functional furniture: kitchen run, bathroom set, study corner, ...

The Poly Haven / MIDI-3D libraries have sofas, beds and shelves but no built-in
joinery.  A flat whose kitchen only gets a plant (the P1 state) is not a home,
so the pieces that make a room *function* are built here as ``builtin:`` parts:
cabinets, a stove, a sink and a fridge for the kitchen; a toilet, a basin and a
shower for the bathroom; a desk, a chair and a dining set; plus the smaller
units (wardrobe, shoe cabinet, storage shelf, washing machine).

Two sides, one contract
-----------------------
**Host side** (pure stdlib -- imported by ``blender_agent/checks.py`` for the
dimension table and by ``layout_gen.py`` for planning): :data:`FIXTURE_PARTS`
maps a part name to its box assembly and to the *measured footprint* that the
boxes span, ``(width_x, depth_y, height_z)`` in metres, un-rotated.  The two are
computed from the same data (:func:`_part` normalises the boxes: XY-centred,
bottom at z = 0), so the collision box of ``checks.py`` is exactly what the
render shows -- a part can never drift from its check.

**Blender side**: :func:`build` turns one part into mesh objects in the
furniture collection, with the conventions of ``details.py`` (world-space
vertices, metres, Z-up, floor at z = 0) and of ``assets.py`` for the yaw
(``rot_z_deg`` rotates about Z, the front faces -Y at 0).  the box list is a
data table, not code, so a part is a few tuples -- and the whole file imports
without ``bpy`` (checks.py and the tests import it on the host).

Usage space
-----------
``usage`` on a part is the clear space its use needs in front of it (fridge
door swing, pulled-out desk chair, shower entry).  ``layout_gen`` keeps that
rectangle free of other blocking furniture -- ``checks.py`` cannot see intent,
so the generator enforces it and reports it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

# box = (suffix, material key, (centre_x, centre_y, bottom_z), (width, depth, height))
Box = Tuple[str, str, Tuple[float, float, float], Tuple[float, float, float]]

# material palette: key -> (name, base colour, roughness, metallic)
MATERIALS: Dict[str, Tuple[str, Tuple[float, float, float], float, float]] = {
    "paint": ("mat_fixture_paint", (0.88, 0.88, 0.86), 0.45, 0.0),
    "counter": ("mat_fixture_counter", (0.24, 0.22, 0.21), 0.32, 0.0),
    "steel": ("mat_fixture_steel", (0.70, 0.71, 0.73), 0.28, 0.9),
    "ceramic": ("mat_fixture_ceramic", (0.94, 0.94, 0.93), 0.16, 0.0),
    "dark": ("mat_fixture_dark", (0.11, 0.11, 0.12), 0.40, 0.1),
    "wood": ("mat_fixture_wood", (0.52, 0.37, 0.24), 0.50, 0.0),
    "fabric": ("mat_fixture_fabric", (0.34, 0.41, 0.49), 0.80, 0.0),
    "glass": ("mat_fixture_glass", (0.85, 0.90, 0.90), 0.04, 0.0),
}


def _part(boxes: Sequence[Box], usage: float = 0.0, category: str = "") -> dict:
    """Normalise a box assembly and derive its measured footprint.

    The boxes are shifted so the union is centred on the local origin in X/Y and
    starts at z = 0 -- the placement convention of ``spec.py``: ``pos`` is the
    centre of the footprint, ``z`` its bottom.  ``dims`` is the size of that
    union, so ``checks.item_dims`` and the geometry cannot disagree.

    Box coordinates: ``(centre_x, centre_y, bottom_z)`` and ``(width, depth,
    height)`` -- the z of a box is its *bottom*, matching the part convention
    (a box sits on what is below it, it is not centred on a height).
    """
    xs = [box[2][0] + sign * box[3][0] / 2.0 for box in boxes for sign in (-1, 1)]
    ys = [box[2][1] + sign * box[3][1] / 2.0 for box in boxes for sign in (-1, 1)]
    z_bottom = min(box[2][2] for box in boxes)
    z_top = max(box[2][2] + box[3][2] for box in boxes)
    shift = (-0.5 * (min(xs) + max(xs)), -0.5 * (min(ys) + max(ys)), -z_bottom)
    moved = [(suffix, material,
              (cx + shift[0], cy + shift[1], cz + shift[2]), (width, depth, height))
             for suffix, material, (cx, cy, cz), (width, depth, height) in boxes]
    dims = (round(max(xs) - min(xs), 4), round(max(ys) - min(ys), 4), round(z_top - z_bottom, 4))
    return {"dims": dims, "boxes": moved, "usage_m": float(usage), "category": category}


# --------------------------------------------------------------------------- #
# kitchen
# --------------------------------------------------------------------------- #
def _kitchen_base() -> List[Box]:
    """Carcass + plinth + door + handle of a 0.60 m base unit (shared by stove/sink)."""
    return [
        ("plinth", "dark", (0.0, 0.0, 0.0), (0.56, 0.54, 0.10)),
        ("carcass", "paint", (0.0, 0.0, 0.10), (0.60, 0.58, 0.78)),
        ("door", "paint", (0.0, -0.30, 0.12), (0.55, 0.03, 0.70)),
        ("handle", "steel", (0.20, -0.315, 0.62), (0.03, 0.03, 0.14)),
    ]


def _kitchen_run_boxes() -> List[Box]:
    return _kitchen_base() + [("worktop", "counter", (0.0, 0.0, 0.88), (0.62, 0.64, 0.04))]


def _kitchen_stove_boxes() -> List[Box]:
    boxes = [
        ("plinth", "dark", (0.0, 0.0, 0.0), (0.56, 0.54, 0.10)),
        ("carcass", "steel", (0.0, 0.0, 0.10), (0.60, 0.58, 0.78)),
        ("oven_door", "dark", (0.0, -0.30, 0.24), (0.54, 0.03, 0.50)),
        ("oven_handle", "steel", (0.0, -0.325, 0.70), (0.54, 0.03, 0.04)),
        ("hob", "dark", (0.0, 0.0, 0.88), (0.60, 0.62, 0.02)),
    ]
    for suffix, (bx, by) in (("fl", (-0.15, -0.16)), ("fr", (0.15, -0.16)),
                             ("bl", (-0.15, 0.16)), ("br", (0.15, 0.16))):
        boxes.append((f"burner_{suffix}", "steel", (bx, by, 0.90), (0.17, 0.17, 0.015)))
    return boxes


def _kitchen_sink_boxes() -> List[Box]:
    return [
        ("plinth", "dark", (0.0, 0.0, 0.0), (0.56, 0.54, 0.10)),
        ("carcass", "paint", (0.0, 0.0, 0.10), (0.60, 0.58, 0.78)),
        ("door", "paint", (0.0, -0.30, 0.12), (0.55, 0.03, 0.70)),
        ("handle", "steel", (0.20, -0.315, 0.62), (0.03, 0.03, 0.14)),
        ("worktop", "counter", (0.0, 0.0, 0.88), (0.62, 0.64, 0.04)),
        ("basin", "ceramic", (0.0, 0.02, 0.86), (0.44, 0.38, 0.10)),
        ("faucet", "steel", (0.0, 0.24, 0.90), (0.05, 0.05, 0.24)),
        ("spout", "steel", (0.0, 0.16, 1.12), (0.04, 0.20, 0.04)),
    ]


def _kitchen_fridge_boxes() -> List[Box]:
    return [
        ("body", "steel", (0.0, 0.0, 0.0), (0.66, 0.68, 1.84)),
        ("freezer_door", "dark", (0.0, -0.35, 0.02), (0.62, 0.03, 0.52)),
        ("fridge_door", "dark", (0.0, -0.35, 0.58), (0.62, 0.03, 1.22)),
        ("freezer_handle", "steel", (0.24, -0.375, 0.26), (0.04, 0.03, 0.22)),
        ("fridge_handle", "steel", (0.24, -0.375, 1.05), (0.04, 0.03, 0.50)),
    ]


def _kitchen_upper_boxes() -> List[Box]:
    return [
        ("carcass", "paint", (0.0, 0.0, 0.0), (0.60, 0.34, 0.70)),
        ("door_l", "paint", (-0.15, -0.185, 0.03), (0.29, 0.03, 0.64)),
        ("door_r", "paint", (0.15, -0.185, 0.03), (0.29, 0.03, 0.64)),
        ("handle_l", "steel", (-0.10, -0.205, 0.32), (0.03, 0.03, 0.12)),
        ("handle_r", "steel", (0.10, -0.205, 0.32), (0.03, 0.03, 0.12)),
    ]


# --------------------------------------------------------------------------- #
# bathroom
# --------------------------------------------------------------------------- #
def _toilet_boxes() -> List[Box]:
    return [
        ("pedestal", "ceramic", (0.0, 0.08, 0.0), (0.26, 0.34, 0.20)),
        ("bowl", "ceramic", (0.0, -0.06, 0.20), (0.36, 0.50, 0.20)),
        ("seat", "ceramic", (0.0, -0.06, 0.40), (0.38, 0.52, 0.04)),
        ("tank", "ceramic", (0.0, 0.26, 0.44), (0.40, 0.16, 0.36)),
        ("lid", "ceramic", (0.0, 0.26, 0.80), (0.42, 0.18, 0.03)),
        ("flush", "steel", (0.12, 0.33, 0.74), (0.05, 0.04, 0.03)),
    ]


def _basin_boxes() -> List[Box]:
    return [
        ("plinth", "dark", (0.0, 0.0, 0.0), (0.50, 0.38, 0.08)),
        ("cabinet", "paint", (0.0, 0.0, 0.08), (0.56, 0.42, 0.66)),
        ("door_l", "paint", (0.0, -0.225, 0.12), (0.27, 0.03, 0.58)),
        ("door_r", "paint", (0.0, -0.225, 0.12), (0.27, 0.03, 0.58)),
        ("handle_l", "steel", (-0.06, -0.245, 0.48), (0.03, 0.03, 0.10)),
        ("handle_r", "steel", (0.06, -0.245, 0.48), (0.03, 0.03, 0.10)),
        ("top", "counter", (0.0, 0.0, 0.74), (0.60, 0.46, 0.04)),
        ("basin", "ceramic", (0.0, -0.02, 0.78), (0.46, 0.34, 0.06)),
        ("faucet", "steel", (0.0, 0.18, 0.78), (0.04, 0.04, 0.26)),
        ("spout", "steel", (0.0, 0.10, 1.00), (0.035, 0.20, 0.035)),
    ]


def _shower_boxes() -> List[Box]:
    return [
        ("tray", "ceramic", (0.0, 0.0, 0.0), (0.92, 0.92, 0.14)),
        ("glass_front", "glass", (0.0, -0.45, 0.14), (0.92, 0.02, 1.86)),
        ("glass_side", "glass", (0.45, 0.0, 0.14), (0.02, 0.92, 1.86)),
        ("frame_front", "steel", (0.0, -0.46, 2.00), (0.94, 0.03, 0.04)),
        ("frame_side", "steel", (0.46, 0.0, 2.00), (0.03, 0.94, 0.04)),
        ("column", "steel", (-0.44, 0.44, 0.14), (0.05, 0.05, 1.90)),
        ("head_arm", "steel", (-0.20, 0.30, 1.92), (0.40, 0.04, 0.04)),
        ("head", "steel", (0.0, 0.30, 1.84), (0.18, 0.18, 0.03)),
    ]


# --------------------------------------------------------------------------- #
# study / dining / bedroom / entry / storage / laundry
# --------------------------------------------------------------------------- #
def _desk_boxes() -> List[Box]:
    return [
        ("top", "wood", (0.0, 0.0, 0.71), (1.20, 0.60, 0.04)),
        ("leg_l", "steel", (-0.58, 0.0, 0.0), (0.04, 0.56, 0.71)),
        ("leg_r", "steel", (0.58, 0.0, 0.0), (0.04, 0.56, 0.71)),
        ("drawers", "paint", (0.34, 0.0, 0.10), (0.42, 0.54, 0.58)),
        ("drawer_1", "paint", (0.34, -0.28, 0.18), (0.38, 0.03, 0.16)),
        ("drawer_2", "paint", (0.34, -0.28, 0.40), (0.38, 0.03, 0.16)),
        ("drawer_handle_1", "steel", (0.34, -0.30, 0.26), (0.16, 0.02, 0.025)),
        ("drawer_handle_2", "steel", (0.34, -0.30, 0.48), (0.16, 0.02, 0.025)),
        ("back_panel", "paint", (-0.20, 0.28, 0.10), (0.72, 0.02, 0.61)),
    ]


def _desk_chair_boxes() -> List[Box]:
    return [
        ("seat", "fabric", (0.0, 0.0, 0.44), (0.48, 0.48, 0.07)),
        ("back", "fabric", (0.0, 0.22, 0.51), (0.46, 0.06, 0.44)),
        ("leg_fl", "steel", (-0.20, -0.20, 0.0), (0.04, 0.04, 0.44)),
        ("leg_fr", "steel", (0.20, -0.20, 0.0), (0.04, 0.04, 0.44)),
        ("leg_bl", "steel", (-0.20, 0.20, 0.0), (0.04, 0.04, 0.44)),
        ("leg_br", "steel", (0.20, 0.20, 0.0), (0.04, 0.04, 0.44)),
    ]


def _dining_table_boxes() -> List[Box]:
    return [
        ("top", "wood", (0.0, 0.0, 0.71), (1.40, 0.80, 0.05)),
        ("leg_fl", "wood", (-0.60, -0.30, 0.0), (0.08, 0.08, 0.71)),
        ("leg_fr", "wood", (0.60, -0.30, 0.0), (0.08, 0.08, 0.71)),
        ("leg_bl", "wood", (-0.60, 0.30, 0.0), (0.08, 0.08, 0.71)),
        ("leg_br", "wood", (0.60, 0.30, 0.0), (0.08, 0.08, 0.71)),
        ("apron", "wood", (0.0, 0.0, 0.62), (1.20, 0.62, 0.06)),
    ]


def _dining_chair_boxes() -> List[Box]:
    return [
        ("seat", "wood", (0.0, 0.0, 0.44), (0.44, 0.44, 0.05)),
        ("back", "wood", (0.0, 0.21, 0.49), (0.44, 0.05, 0.43)),
        ("leg_fl", "wood", (-0.19, -0.19, 0.0), (0.04, 0.04, 0.44)),
        ("leg_fr", "wood", (0.19, -0.19, 0.0), (0.04, 0.04, 0.44)),
        ("leg_bl", "wood", (-0.19, 0.19, 0.0), (0.04, 0.04, 0.44)),
        ("leg_br", "wood", (0.19, 0.19, 0.0), (0.04, 0.04, 0.44)),
    ]


def _wardrobe_boxes() -> List[Box]:
    return [
        ("plinth", "dark", (0.0, 0.0, 0.0), (1.14, 0.56, 0.08)),
        ("carcass", "paint", (0.0, 0.0, 0.08), (1.20, 0.60, 2.02)),
        ("door_l", "paint", (-0.295, -0.315, 0.10), (0.58, 0.03, 1.96)),
        ("door_r", "paint", (0.295, -0.315, 0.10), (0.58, 0.03, 1.96)),
        ("handle_l", "steel", (-0.04, -0.34, 1.02), (0.03, 0.03, 0.26)),
        ("handle_r", "steel", (0.04, -0.34, 1.02), (0.03, 0.03, 0.26)),
    ]


def _shoe_cabinet_boxes() -> List[Box]:
    return [
        ("plinth", "dark", (0.0, 0.0, 0.0), (0.84, 0.32, 0.08)),
        ("carcass", "paint", (0.0, 0.0, 0.08), (0.90, 0.36, 0.98)),
        ("door_l", "paint", (-0.22, -0.185, 0.10), (0.42, 0.03, 0.94)),
        ("door_r", "paint", (0.22, -0.185, 0.10), (0.42, 0.03, 0.94)),
        ("handle_l", "steel", (-0.02, -0.21, 0.55), (0.03, 0.03, 0.20)),
        ("handle_r", "steel", (0.02, -0.21, 0.55), (0.03, 0.03, 0.20)),
    ]


def _storage_shelf_boxes() -> List[Box]:
    boxes = [
        ("side_l", "wood", (-0.43, 0.0, 0.0), (0.04, 0.40, 1.80)),
        ("side_r", "wood", (0.43, 0.0, 0.0), (0.04, 0.40, 1.80)),
        ("back", "wood", (0.0, 0.19, 0.0), (0.82, 0.02, 1.80)),
    ]
    for index, z in enumerate((0.10, 0.52, 0.94, 1.36, 1.74)):
        boxes.append((f"shelf_{index}", "wood", (0.0, 0.0, z), (0.82, 0.38, 0.03)))
    return boxes


def _washer_boxes() -> List[Box]:
    boxes = [
        ("body", "paint", (0.0, 0.0, 0.0), (0.60, 0.60, 0.84)),
        ("control", "dark", (0.0, -0.29, 0.66), (0.56, 0.03, 0.16)),
        ("knob", "steel", (0.20, -0.31, 0.74), (0.05, 0.03, 0.05)),
        ("door_ring", "steel", (0.0, -0.315, 0.34), (0.46, 0.03, 0.46)),
    ]
    return boxes


FIXTURE_PARTS: Dict[str, dict] = {
    # kitchen
    "kitchen_cabinet": _part(_kitchen_run_boxes(), usage=0.0, category="kitchen"),
    "kitchen_stove": _part(_kitchen_stove_boxes(), usage=0.60, category="kitchen"),
    "kitchen_sink": _part(_kitchen_sink_boxes(), usage=0.60, category="kitchen"),
    "kitchen_fridge": _part(_kitchen_fridge_boxes(), usage=0.75, category="kitchen"),
    "kitchen_upper": _part(_kitchen_upper_boxes(), usage=0.0, category="kitchen"),
    # bathroom
    "bath_toilet": _part(_toilet_boxes(), usage=0.55, category="bathroom"),
    "bath_basin": _part(_basin_boxes(), usage=0.55, category="bathroom"),
    "bath_shower": _part(_shower_boxes(), usage=0.60, category="bathroom"),
    # study / dining / bedroom / entry / storage / laundry
    "study_desk": _part(_desk_boxes(), usage=0.75, category="study"),
    "study_chair": _part(_desk_chair_boxes(), usage=0.55, category="study"),
    "dining_table": _part(_dining_table_boxes(), usage=0.0, category="dining"),
    "dining_chair": _part(_dining_chair_boxes(), usage=0.55, category="dining"),
    "wardrobe": _part(_wardrobe_boxes(), usage=0.65, category="bedroom"),
    "entry_shoe_cabinet": _part(_shoe_cabinet_boxes(), usage=0.55, category="entry"),
    "storage_shelf": _part(_storage_shelf_boxes(), usage=0.60, category="storage"),
    "laundry_washer": _part(_washer_boxes(), usage=0.65, category="laundry"),
}

#: ``builtin:<part>`` -> footprint, the table ``checks.item_dims`` needs.
BUILTIN_DIMS_M: Dict[str, Tuple[float, float, float]] = {
    f"builtin:{name}": part["dims"] for name, part in FIXTURE_PARTS.items()
}


def dims(part: str) -> Tuple[float, float, float]:
    """Measured ``(width, depth, height)`` of a fixture part, un-rotated."""
    return FIXTURE_PARTS[part]["dims"]


def usage_m(part: str) -> float:
    """Clear space the part needs in front of it (door swing / pull-out)."""
    return float(FIXTURE_PARTS[part]["usage_m"])


def is_fixture(kind: str) -> bool:
    return kind in FIXTURE_PARTS


# --------------------------------------------------------------------------- #
# Blender side
# --------------------------------------------------------------------------- #
def _material(key: str):
    import bpy

    name, color, roughness, metallic = MATERIALS[key]
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        bsdf = material.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Roughness"].default_value = float(roughness)
        bsdf.inputs["Metallic"].default_value = float(metallic)
        if key == "glass":
            bsdf.inputs["Transmission Weight"].default_value = 1.0
            bsdf.inputs["IOR"].default_value = 1.45
    return material


def build(kind: str, pos: Sequence[float], rot_z_deg: float, collection,
          name: Optional[str] = None, base_z: float = 0.0) -> List:
    """Build one fixture part as mesh objects in ``collection``.

    ``pos`` is the centre of the part's footprint (the spec convention),
    ``rot_z_deg`` rotates about Z with the front facing -Y at 0 (the asset
    convention of ``assets.py``), ``base_z`` lifts the whole assembly (a
    wall-mounted upper cabinet).  Returns the created objects, bottom face at
    ``base_z``.
    """
    from blender_agent import shell as shell_mod

    part = FIXTURE_PARTS[kind]
    label = name or f"fixture_{kind}"
    theta = math.radians(float(rot_z_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    direction = (cos_t, sin_t)
    objects = []
    for suffix, material_key, (cx, cy, cz), (width, depth, height) in part["boxes"]:
        wx = float(pos[0]) + cos_t * cx - sin_t * cy
        wy = float(pos[1]) + sin_t * cx + cos_t * cy
        if width <= 0.0 or depth <= 0.0 or height <= 0.0:
            continue
        obj = shell_mod.add_oriented_box(
            f"{label}_{suffix}", (wx, wy), direction, width, depth,
            float(base_z) + cz, float(base_z) + cz + height, collection,
        )
        obj.data.materials.append(_material(material_key))
        objects.append(obj)
    return objects
