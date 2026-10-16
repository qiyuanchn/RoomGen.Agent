"""Room-shell geometry: floors, ceilings, walls and door/window cutters.

The shell is built from :mod:`blender_agent.spec` with plain mesh data (no
``bpy.ops`` primitive helpers, which are fragile in ``--background`` runs):

* one floor prism per room, spanning ``z in [-0.05, 0]``
* one ceiling prism per room, spanning ``z in [wall_height, wall_height + 0.05]``
* one wall box per merged wall span (see ``spec.build_wall_segments``), the
  polygon centre line carrying a ``wall_thickness`` box that is extended by
  half a thickness at both ends so perpendicular walls close the corners
* door/window openings are boolean-differenced out of the walls only, so a
  cut below ``z = 0`` cannot notch the floor slabs
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import bpy

from blender_agent import spec as spec_mod

SHELL_COLLECTION = "shell"
CUTTER_COLLECTION = "cutters"

FLOOR_SLAB = 0.05
CEILING_SLAB = 0.05


def ensure_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def link_object(name: str, mesh: bpy.types.Mesh, collection: bpy.types.Collection) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def add_prism(
    name: str,
    polygon_xy: Sequence[Sequence[float]],
    z_bottom: float,
    z_top: float,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Closed prism from an XY polygon (auto-oriented to counter-clockwise)."""
    polygon = [(float(p[0]), float(p[1])) for p in polygon_xy]
    area = spec_mod.polygon_area(polygon)
    if area < 0.0:
        polygon.reverse()
    elif abs(area) < 1e-9:
        raise ValueError(f"degenerate polygon for '{name}'")

    n = len(polygon)
    bottom = float(min(z_bottom, z_top))
    top = float(max(z_bottom, z_top))
    verts: List[Tuple[float, float, float]] = [(x, y, top) for x, y in polygon]
    verts += [(x, y, bottom) for x, y in polygon]

    faces: List[List[int]] = [list(range(n))]  # top face, CCW as given
    faces.append(list(range(2 * n - 1, n - 1, -1)))  # bottom face, reversed
    for i in range(n):
        j = (i + 1) % n
        faces.append([j, i, i + n, j + n])  # outward side quad

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    return link_object(name, mesh, collection)


def add_oriented_box(
    name: str,
    center_xy: Sequence[float],
    direction: Sequence[float],
    length: float,
    depth: float,
    z_bottom: float,
    z_top: float,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Box whose ``length`` runs along ``direction`` and ``depth`` across it."""
    dx, dy = float(direction[0]), float(direction[1])
    norm = math.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm
    nx, ny = -dy, dx
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half_l, half_d = length / 2.0, depth / 2.0
    polygon = [
        (cx - dx * half_l + nx * half_d, cy - dy * half_l + ny * half_d),
        (cx + dx * half_l + nx * half_d, cy + dy * half_l + ny * half_d),
        (cx + dx * half_l - nx * half_d, cy + dy * half_l - ny * half_d),
        (cx - dx * half_l - nx * half_d, cy - dy * half_l - ny * half_d),
    ]
    return add_prism(name, polygon, z_bottom, z_top, collection)


def _world_bbox(obj: bpy.types.Object) -> Tuple[float, float, float, float]:
    xs = [(obj.matrix_world @ v.co).x for v in obj.data.vertices]
    ys = [(obj.matrix_world @ v.co).y for v in obj.data.vertices]
    return min(xs), min(ys), max(xs), max(ys)


def build_cutters(spec: spec_mod.SceneSpec, collection: bpy.types.Collection) -> List[bpy.types.Object]:
    """One box per door/window opening, sized to punch through the wall."""
    try:
        segments = spec.wall_segments()
    except spec_mod.SpecError:
        segments = []
    thickness = spec.wall_thickness
    depth = thickness + 0.6
    cutters: List[bpy.types.Object] = []

    for index, opening in enumerate(spec.openings):
        segment = spec_mod.wall_segment_for_point(segments, opening.center, opening.between[0])
        direction = segment.direction if segment is not None else (1.0, 0.0)
        cutters.append(
            add_oriented_box(
                name=f"cutter_opening_{index}",
                center_xy=opening.center,
                direction=direction,
                length=opening.width,
                depth=depth,
                z_bottom=-0.2,
                z_top=opening.height,
                collection=collection,
            )
        )

    for index, window in enumerate(spec.windows):
        x0, y0 = window.wall[0]
        x1, y1 = window.wall[1]
        length = math.hypot(x1 - x0, y1 - y0)
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        cutters.append(
            add_oriented_box(
                name=f"cutter_window_{index}",
                center_xy=center,
                direction=(x1 - x0, y1 - y0),
                length=length,
                depth=depth,
                z_bottom=window.sill,
                z_top=window.sill + window.height,
                collection=collection,
            )
        )
    return cutters


def cut_walls(walls: Sequence[bpy.types.Object], cutters: Sequence[bpy.types.Object]) -> Dict[str, int]:
    """Boolean-difference every cutter out of every wall it overlaps, then apply."""
    view_layer = bpy.context.view_layer
    applied: Dict[str, int] = {}
    for wall in walls:
        wx0, wy0, wx1, wy1 = _world_bbox(wall)
        hits = 0
        for cutter in cutters:
            cx0, cy0, cx1, cy1 = _world_bbox(cutter)
            if cx1 < wx0 or cx0 > wx1 or cy1 < wy0 or cy0 > wy1:
                continue
            modifier = wall.modifiers.new(name=f"cut_{cutter.name}", type="BOOLEAN")
            modifier.operation = "DIFFERENCE"
            modifier.solver = "EXACT"
            modifier.object = cutter
            for obj in bpy.context.view_layer.objects:
                obj.select_set(False)
            wall.select_set(True)
            view_layer.objects.active = wall
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            hits += 1
        applied[wall.name] = hits
    return applied


def delete_objects(objects: Iterable[bpy.types.Object]) -> None:
    for obj in list(objects):
        mesh = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def build_shell(spec: spec_mod.SceneSpec, collection_name: str = SHELL_COLLECTION) -> dict:
    """Build floors, ceilings and walls; openings are cut before returning."""
    collection = ensure_collection(collection_name)
    thickness = spec.wall_thickness
    height = spec.wall_height

    floors = []
    ceilings = []
    for room in spec.rooms:
        floors.append(
            add_prism(f"floor_{room.id}", room.polygon, -FLOOR_SLAB, 0.0, collection)
        )
        ceilings.append(
            add_prism(f"ceiling_{room.id}", room.polygon, height, height + CEILING_SLAB, collection)
        )

    walls = []
    for index, segment in enumerate(spec.wall_segments()):
        label = "_".join(segment.owners) or "wall"
        walls.append(
            add_oriented_box(
                name=f"wall_{index:02d}_{label}_{segment.p0[0]:.2f}_{segment.p0[1]:.2f}".replace("-", "m"),
                center_xy=((segment.p0[0] + segment.p1[0]) / 2.0, (segment.p0[1] + segment.p1[1]) / 2.0),
                direction=segment.direction,
                length=segment.length + thickness,
                depth=thickness,
                z_bottom=0.0,
                z_top=height,
                collection=collection,
            )
        )

    cutter_collection = ensure_collection(CUTTER_COLLECTION)
    cutters = build_cutters(spec, cutter_collection)
    applied = cut_walls(walls, cutters)
    delete_objects(cutters)
    if bpy.data.collections.get(CUTTER_COLLECTION) is not None and not bpy.data.collections[CUTTER_COLLECTION].objects:
        bpy.data.collections.remove(bpy.data.collections[CUTTER_COLLECTION])

    return {
        "collection": collection,
        "floors": floors,
        "ceilings": ceilings,
        "walls": walls,
        "shell_objects": floors + ceilings + walls,
        "cutters_applied": applied,
    }
