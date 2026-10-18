"""Room-shell geometry: floors, ceilings, walls and door/window cutters.

The shell is built from :mod:`blender_agent.spec` with plain mesh data (no
``bpy.ops`` primitive helpers, which are fragile in ``--background`` runs):

* one floor slab prism per room, spanning ``z in [elevation - slab_thickness,
  elevation]`` (legacy single-storey specs: ``z in [-0.05, 0]``)
* a ceiling/roof prism for exactly the part of a room that no other storey's
  slab covers (``spec.ceiling_regions``): a fully covered room gets none -- the
  storey above *is* the ceiling -- a partially covered one gets one prism per
  surviving region, so the upper slab and the lower ceiling are never one and the
  same duplicated body and never leave a hole either
* one wall box per merged wall span (see ``spec.build_wall_segments``), the
  polygon centre line carrying a ``wall_thickness`` box that is extended by
  half a thickness at both ends so perpendicular walls close the corners, plus
  ``WALL_END_OVERLAP_M`` so the two end faces of a concave corner are not
  coplanar (Cycles renders that z-fighting as a black stripe); spans merge
  within one storey only and every wall box lives in its storey's z band
* door/window openings are boolean-differenced out of the walls only, so a
  cut below the floor surface cannot notch the floor slabs; a cutter only
  touches the wall boxes whose z band it overlaps (a second-storey door never
  cuts a first-storey wall)
* ``slab_openings`` punch their outline through the floor slab of every listed
  storey (stairwell / void) and ``kind: "stair"`` / ``"ramp"`` connectors are
  built as real steps / wedges (see ``spec.stair_plan`` / ``spec.ramp_plan``)
* a declared **void** cuts the outline out of every storey above its lower
  floor and the lower room loses that ceiling area; the guard rails of
  ``spec.railing_boxes`` are built as top rail, mid rail and posts
* a split-level room (``rooms[].level``) gets a solid plinth up to its raised
  walking surface, its ceiling rises with it, wall spans run from the lowest to
  the highest room they bound, and a same-storey ramp gets a passage cut in the
  wall at its head
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import bpy

from blender_agent import spec as spec_mod

SHELL_COLLECTION = "shell"
CUTTER_COLLECTION = "cutters"

FLOOR_SLAB = spec_mod.DEFAULT_SLAB_M
CEILING_SLAB = spec_mod.DEFAULT_SLAB_M
#: extra wall-box length per end, past the half-thickness corner closure.  Two
#: perpendicular walls would otherwise put two faces in the same plane and
#: Cycles renders the coincident strip black.
WALL_END_OVERLAP_M = 0.002

STAIR_COLOR = (0.62, 0.47, 0.33)
RAILING_COLOR = (0.35, 0.33, 0.30)


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


def _world_bbox(obj: bpy.types.Object) -> Tuple[float, float, float, float, float, float]:
    """World-space AABB ``(x0, y0, z0, x1, y1, z1)`` of a mesh object."""
    xs, ys, zs = [], [], []
    for vertex in obj.data.vertices:
        point = obj.matrix_world @ vertex.co
        xs.append(point.x)
        ys.append(point.y)
        zs.append(point.z)
    return min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)


def build_cutters(spec: spec_mod.SceneSpec, collection: bpy.types.Collection) -> List[bpy.types.Object]:
    """One box per door/window opening, sized to punch through its own storey's wall.

    Doors come from ``spec.door_connections()`` (parsed connectors plus the
    legacy ``openings``), so a multi-floor spec cuts a door out of the wall of
    the storey the two rooms live on: the cutter spans ``[elevation - 0.2,
    elevation + height]``.  A window sill/height is relative to the room floor,
    so its cutter spans ``[elevation + sill, elevation + sill + height]``.
    """
    try:
        segments = spec.wall_segments()
    except spec_mod.SpecError:
        segments = []
    thickness = spec.wall_thickness
    depth = thickness + 0.6
    cutters: List[bpy.types.Object] = []

    for index, connector in enumerate(spec.door_connections()):
        floor = spec.room_floor(connector.from_room)
        center = connector.center
        segment = spec_mod.wall_segment_for_point(
            segments, center, connector.from_room,
            floor_id=connector.from_floor if spec.floors else None,
        )
        direction = segment.direction if segment is not None else (1.0, 0.0)
        height = float(connector.height or spec_mod.DEFAULT_DOOR_HEIGHT_M)
        elevation = spec.room_elevation(connector.from_room)
        cutters.append(
            add_oriented_box(
                name=f"cutter_{connector.kind}_{index}",
                center_xy=center,
                direction=direction,
                length=connector.width,
                depth=depth,
                z_bottom=elevation - 0.2,
                z_top=elevation + height,
                collection=collection,
            )
        )

    for index, window in enumerate(spec.windows):
        floor = spec.room_floor(window.room)
        elevation = spec.room_elevation(window.room)
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
                z_bottom=elevation + window.sill,
                z_top=elevation + window.sill + window.height,
                collection=collection,
            )
        )

    # a split-level ramp stays in the lower room; the shared wall needs a
    # passage from the lower floor up to the raised walking surface
    for index, plan in enumerate(spec.ramp_plans()):
        connector = plan.connector
        if connector.from_floor != connector.to_floor:
            continue
        if abs(plan.to_elevation - plan.from_elevation) < 1e-6:
            continue
        wall_point = (plan.center[0] + plan.direction[0] * plan.run / 2.0,
                      plan.center[1] + plan.direction[1] * plan.run / 2.0)
        segment = spec_mod.wall_segment_for_point(
            segments, wall_point, connector.from_room, floor_id=connector.from_floor)
        if segment is None:
            segment = spec_mod.wall_segment_for_point(
                segments, wall_point, connector.to_room, floor_id=connector.to_floor)
        if segment is None:
            continue
        from_room = spec.room(connector.from_room)
        to_room = spec.room(connector.to_room)
        z_bottom = min(spec.room_elevation(from_room.id), spec.room_elevation(to_room.id)) - 0.2
        z_top = min(spec.room_top(from_room.id), spec.room_top(to_room.id))
        cutters.append(
            add_oriented_box(
                name=f"cutter_split_level_{index}",
                center_xy=wall_point,
                direction=segment.direction,
                length=plan.width + 0.10,
                depth=depth,
                z_bottom=z_bottom,
                z_top=z_top,
                collection=collection,
            )
        )
    return cutters


def cut_walls(walls: Sequence[bpy.types.Object], cutters: Sequence[bpy.types.Object]) -> Dict[str, int]:
    """Boolean-difference every cutter out of every wall it overlaps, then apply.

    The overlap test is three-dimensional: on a multi-floor plan the walls of two
    storeys share their XY footprint, and only the cutter's own z band may bite.
    """
    view_layer = bpy.context.view_layer
    applied: Dict[str, int] = {}
    for wall in walls:
        wx0, wy0, wz0, wx1, wy1, wz1 = _world_bbox(wall)
        hits = 0
        for cutter in cutters:
            cx0, cy0, cz0, cx1, cy1, cz1 = _world_bbox(cutter)
            if cx1 < wx0 or cx0 > wx1 or cy1 < wy0 or cy0 > wy1 or cz1 < wz0 or cz0 > wz1:
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


def _stair_material() -> bpy.types.Material:
    """Flat wood-tone material for stairs/ramps (they are shell, not furniture)."""
    material = bpy.data.materials.get("mat_stair")
    if material is None:
        material = bpy.data.materials.new("mat_stair")
        material.use_nodes = True
        bsdf = material.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (STAIR_COLOR[0], STAIR_COLOR[1], STAIR_COLOR[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6
    return material


def _railing_material() -> bpy.types.Material:
    material = bpy.data.materials.get("mat_railing")
    if material is None:
        material = bpy.data.materials.new("mat_railing")
        material.use_nodes = True
        bsdf = material.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (RAILING_COLOR[0], RAILING_COLOR[1],
                                                   RAILING_COLOR[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.45
        bsdf.inputs["Metallic"].default_value = 0.25
    return material


def add_wedge(name: str, bottom: Sequence[Sequence[float]], top: Sequence[Sequence[float]],
              collection: bpy.types.Collection) -> bpy.types.Object:
    """Solid between two matching quads (bottom and a sloped top face)."""
    count = len(bottom)
    verts = [(float(p[0]), float(p[1]), float(p[2])) for p in bottom] \
        + [(float(p[0]), float(p[1]), float(p[2])) for p in top]
    faces: List[List[int]] = [list(range(count)), list(range(2 * count - 1, count - 1, -1))]
    for i in range(count):
        j = (i + 1) % count
        faces.append([i, j, j + count, i + count])
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    return link_object(name, mesh, collection)


def _synthetic_floor(spec: spec_mod.SceneSpec) -> spec_mod.FloorSpec:
    return spec_mod.FloorSpec(id="", elevation=0.0, height=spec.wall_height,
                              slab_thickness=FLOOR_SLAB)


def build_stairs(spec: spec_mod.SceneSpec, collection: bpy.types.Collection) -> dict:
    """Build every stair step / landing box and every ramp wedge."""
    stairs: List[bpy.types.Object] = []
    ramps: List[bpy.types.Object] = []
    material = None
    for plan in spec.stair_plans():
        label = f"{plan.connector.from_room}_{plan.connector.to_room}"
        for index, box in enumerate(plan.boxes):
            obj = add_prism(f"stair_{label}_{box['role']}_{index:02d}", box["polygon"],
                            box["z0"], box["z1"], collection)
            material = material or _stair_material()
            obj.data.materials.append(material)
            stairs.append(obj)
    for plan in spec.ramp_plans():
        label = f"{plan.connector.from_room}_{plan.connector.to_room}"
        obj = add_wedge(f"ramp_{label}", plan.wedge["bottom"], plan.wedge["top"], collection)
        material = material or _stair_material()
        obj.data.materials.append(material)
        ramps.append(obj)
    return {"stairs": stairs, "ramps": ramps}


def build_railings(spec: spec_mod.SceneSpec, collection: bpy.types.Collection) -> List[bpy.types.Object]:
    """Guard rails of the declared voids: top rail, mid rail and posts."""
    railings: List[bpy.types.Object] = []
    material = None
    for entry in spec.railing_boxes():
        obj = add_prism(entry["name"], entry["polygon"], entry["z0"], entry["z1"], collection)
        material = material or _railing_material()
        obj.data.materials.append(material)
        railings.append(obj)
    return railings


def build_shell(spec: spec_mod.SceneSpec, collection_name: str = SHELL_COLLECTION) -> dict:
    """Build floors, ceilings, walls, stairs and slab openings per storey."""
    collection = ensure_collection(collection_name)
    thickness = spec.wall_thickness
    storeys = spec.floors or [_synthetic_floor(spec)]

    floors: List[bpy.types.Object] = []
    ceilings: List[bpy.types.Object] = []
    walls: List[bpy.types.Object] = []
    storey_report: List[dict] = []
    covered_by_upper: List[str] = []
    slabs_by_floor: Dict[str, List[bpy.types.Object]] = {}

    for floor in storeys:
        rooms = spec.floor_rooms(floor.id) if spec.floors else list(spec.rooms)
        height = floor.height
        floor_objects: List[bpy.types.Object] = []
        for room in rooms:
            elevation = spec.room_elevation(room.id)
            # a split-level room's slab reaches down to the storey slab bottom:
            # the raised floor is a solid plinth, not a floating plate
            floor_objects.append(
                add_prism(f"floor_{room.id}", room.polygon,
                          floor.elevation - floor.slab_thickness, elevation, collection)
            )
        floors.extend(floor_objects)
        slabs_by_floor[floor.id] = floor_objects

        ceiling_objects: List[bpy.types.Object] = []
        for room in rooms:
            top = spec.room_top(room.id)
            parts = spec.ceiling_regions(room.id)
            replaced = spec.ceiling_replaced_by(room.id)
            if not parts:
                upper = ", ".join(sorted({cover["floor"] for cover in replaced}))
                covered_by_upper.append(f"ceiling_{room.id} (slab of floor '{upper}')")
                continue
            for index, polygon in enumerate(parts):
                name = f"ceiling_{room.id}" if len(parts) == 1 else f"ceiling_{room.id}_{index}"
                ceiling_objects.append(
                    add_prism(name, polygon, top, top + floor.slab_thickness, collection)
                )
            if replaced:
                area = sum(cover["area_m2"] for cover in replaced)
                covered_by_upper.append(
                    f"ceiling_{room.id} (partly under floor "
                    f"'{'/'.join(sorted({cover['floor'] for cover in replaced}))}', "
                    f"{area:.2f} m^2 replaced, {len(parts)} part(s) built)"
                )
        ceilings.extend(ceiling_objects)

        segments = spec_mod.build_wall_segments(rooms)
        wall_objects = []
        for index, segment in enumerate(segments):
            label = "_".join(segment.owners) or "wall"
            owner_rooms = [spec.room(owner) for owner in segment.owners]
            z_bottom = min((spec.room_elevation(room.id) for room in owner_rooms),
                           default=floor.elevation)
            z_top = max((spec.room_top(room.id) for room in owner_rooms),
                        default=floor.elevation + height)
            wall_objects.append(
                add_oriented_box(
                    name=f"wall_{index:02d}_{label}_{segment.p0[0]:.2f}_{segment.p0[1]:.2f}".replace("-", "m"),
                    center_xy=((segment.p0[0] + segment.p1[0]) / 2.0, (segment.p0[1] + segment.p1[1]) / 2.0),
                    direction=segment.direction,
                    # Half a thickness at each end closes a corner, and the
                    # extra millimetres keep the two end faces from being
                    # coplanar with the perpendicular wall's side face: that
                    # z-fighting renders as a black stripe at concave corners.
                    length=segment.length + thickness + 2.0 * WALL_END_OVERLAP_M,
                    depth=thickness,
                    z_bottom=z_bottom,
                    z_top=z_top,
                    collection=collection,
                )
            )
        walls.extend(wall_objects)
        storey_report.append({
            "id": floor.id, "elevation": floor.elevation, "height": height,
            "slab_thickness": floor.slab_thickness, "rooms": [room.id for room in rooms],
            "room_levels": {room.id: round(spec.room_elevation(room.id), 4) for room in rooms},
            "walls": len(wall_objects), "floor_slabs": [obj.name for obj in floor_objects],
            "ceilings": [obj.name for obj in ceiling_objects],
        })

    # -- slab openings: punch the outline through the listed storeys' slabs --- #
    slab_report: List[dict] = []
    if spec.slab_openings:
        cutter_collection = ensure_collection(CUTTER_COLLECTION)
        for index, opening in enumerate(spec.slab_openings):
            for floor_id in opening.floors:
                floor = spec.floor(floor_id)
                slabs = slabs_by_floor.get(floor_id, [])
                prism = add_prism(
                    f"cutter_slab_{index}_{floor_id}", opening.outline,
                    floor.elevation - floor.slab_thickness - 0.05, floor.elevation + 0.05,
                    cutter_collection,
                )
                applied = cut_walls(slabs, [prism])
                delete_objects([prism])
                slab_report.append({
                    "index": index, "floor": floor_id, "slabs": [obj.name for obj in slabs],
                    "applied": sum(applied.values()),
                })

    # -- declared voids: cut every storey above the void's own floor --------- #
    void_report: List[dict] = []
    if spec.voids:
        cutter_collection = ensure_collection(CUTTER_COLLECTION)
        for index, void in enumerate(spec.voids):
            lower = spec.floor(void.floor)
            for floor in spec.floors:
                if floor.elevation <= lower.elevation + 1e-9:
                    continue
                slabs = slabs_by_floor.get(floor.id, [])
                if not slabs:
                    continue
                prism = add_prism(
                    f"cutter_void_{index}_{floor.id}", void.outline,
                    floor.elevation - floor.slab_thickness - 0.05, floor.elevation + 0.05,
                    cutter_collection,
                )
                applied = cut_walls(slabs, [prism])
                delete_objects([prism])
                void_report.append({
                    "index": index, "name": void.name or f"void_{index}",
                    "floor": floor.id, "slabs": [obj.name for obj in slabs],
                    "applied": sum(applied.values()),
                })

    # -- stairs / ramps ------------------------------------------------------ #
    stair_info = build_stairs(spec, collection)
    railing_objects = build_railings(spec, collection)

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
        "stairs": stair_info["stairs"],
        "ramps": stair_info["ramps"],
        "railings": railing_objects,
        "shell_objects": floors + ceilings + walls + stair_info["stairs"] + stair_info["ramps"]
        + railing_objects,
        "cutters_applied": applied,
        "storeys": storey_report,
        "ceiling_from_upper": covered_by_upper,
        "slab_openings": slab_report,
        "voids": void_report,
    }
