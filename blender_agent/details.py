"""Realism details on top of the P0 shell: skirting boards, door/window casings,
window sills (and optional glass), shell edge bevels plus a few small props
(rug, table lamp) that live in the same collections the build exports.

Everything is built from plain mesh data through :func:`shell.add_prism` /
:func:`shell.add_oriented_box` so it shares the shell's conventions: world-space
vertices, Z-up, metres, floor at ``z = 0``.

Geometry rules
--------------
skirting   follows each room polygon edge on the *inner* wall face
           (``wall_thickness / 2`` inside the boundary), 80 mm high, 14 mm
           thick, clipped out of door openings and covered by the casings.
casings    door/window reveals get a 60 mm casing on both wall faces, standing
           ``frame_depth`` proud of the wall; windows add a sill board.
bevels     a 2.5 mm bevel modifier on shell objects so sunlight catches the
           edges instead of showing razor-sharp CG corners.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import bpy

from blender_agent import shell as shell_mod
from blender_agent import spec as spec_mod

SKIRTING_HEIGHT = 0.08
SKIRTING_THICKNESS = 0.014
CASING_WIDTH = 0.06
CASING_DEPTH = 0.022
SILL_THICKNESS = 0.024
SILL_OVERHANG = 0.035
GLASS_THICKNESS = 0.004
BEVEL_WIDTH = 0.0025
BEVEL_SEGMENTS = 2


def _subtract(span: Tuple[float, float], cuts: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """1-D interval subtraction."""
    parts = [span]
    for cut_start, cut_end in cuts:
        out = []
        for start, end in parts:
            if cut_end <= start or cut_start >= end:
                out.append((start, end))
                continue
            if cut_start > start:
                out.append((start, cut_start))
            if cut_end < end:
                out.append((cut_end, end))
        parts = out
    return [(a, b) for a, b in parts if b - a > 1e-4]


def _opening_cuts(spec: spec_mod.SceneSpec, segments) -> Dict[tuple, List[Tuple[float, float]]]:
    """opening intervals in each wall line's parameter space (keyed like spec's)."""
    cuts: Dict[tuple, List[Tuple[float, float]]] = {}
    for opening in spec.openings:
        segment = spec_mod.wall_segment_for_point(segments, opening.center, opening.between[0])
        if segment is None:
            continue
        key = spec_mod.canonical_line_key(segment.p0, segment.p1)
        base = spec_mod.line_parameter(segment.p0, segment.direction)
        centre = spec_mod.line_parameter(opening.center, segment.direction)
        half = opening.width / 2.0 + CASING_WIDTH
        cuts.setdefault(key, []).append((centre - half - base, centre + half - base))
    return cuts


def build_skirting(spec: spec_mod.SceneSpec, collection, thickness: float = SKIRTING_THICKNESS,
                   height: float = SKIRTING_HEIGHT) -> List:
    """Skirting boards hugging every room wall, clipped at door openings."""
    segments = spec.wall_segments()
    cuts = _opening_cuts(spec, segments)
    objects = []
    for room in spec.rooms:
        polygon = room.polygon
        count = len(polygon)
        for index in range(count):
            ax, ay = float(polygon[index][0]), float(polygon[index][1])
            bx, by = float(polygon[(index + 1) % count][0]), float(polygon[(index + 1) % count][1])
            length = math.hypot(bx - ax, by - ay)
            if length < 1e-6:
                continue
            ux, uy = (bx - ax) / length, (by - ay) / length
            # counter-clockwise polygons: the inward normal is left of the edge
            nx, ny = -uy, ux
            key, (cux, cuy), (cnx, cny), cc = spec_mod.canonical_line((ax, ay), (bx, by))
            base = spec_mod.line_parameter((ax, ay), (cux, cuy))
            local_cuts = [(a - base, b - base) for a, b in cuts.get(key, [])]
            for start, end in _subtract((0.0, length), local_cuts):
                mid = (start + end) / 2.0
                centre = (
                    ax + ux * mid + nx * (spec.wall_thickness / 2.0 + thickness / 2.0),
                    ay + uy * mid + ny * (spec.wall_thickness / 2.0 + thickness / 2.0),
                )
                objects.append(
                    shell_mod.add_oriented_box(
                        name=f"skirting_{room.id}_{index:02d}_{int(start * 1000):04d}",
                        center_xy=centre,
                        direction=(ux, uy),
                        length=end - start,
                        depth=thickness,
                        z_bottom=0.0,
                        z_top=height,
                        collection=collection,
                    )
                )
    return objects


def build_casings(spec: spec_mod.SceneSpec, collection) -> List:
    """Door + window casings and window sills."""
    segments = spec.wall_segments()
    objects = []
    for index, opening in enumerate(spec.openings):
        segment = spec_mod.wall_segment_for_point(segments, opening.center, opening.between[0])
        if segment is None:
            raise spec_mod.SpecError(f"opening {index}: wall segment not found")
        direction = segment.direction
        depth = spec.wall_thickness + 2.0 * CASING_DEPTH
        centre = (float(opening.center[0]), float(opening.center[1]))
        # vertical jambs, one per side of the opening
        for side, sign in (("a", -1), ("b", 1)):
            offset = sign * (opening.width / 2.0 + CASING_WIDTH / 2.0)
            centre_i = (centre[0] + direction[0] * offset, centre[1] + direction[1] * offset)
            objects.append(
                shell_mod.add_oriented_box(
                    f"{opening.kind}_casing_{index}_{side}", centre_i, direction, CASING_WIDTH, depth,
                    0.0 if opening.kind == "door" else opening.height, opening.height + CASING_WIDTH * 0.0,
                    collection,
                )
            )
        # head casing
        head_z0 = opening.height
        head_z1 = opening.height + CASING_WIDTH
        objects.append(
            shell_mod.add_oriented_box(
                f"{opening.kind}_casing_{index}_head", centre, direction, opening.width + 2 * CASING_WIDTH,
                depth, head_z0, head_z1, collection,
            )
        )
    for index, window in enumerate(spec.windows):
        p0 = (float(window.wall[0][0]), float(window.wall[0][1]))
        p1 = (float(window.wall[1][0]), float(window.wall[1][1]))
        length = math.dist(p0, p1)
        direction = ((p1[0] - p0[0]) / length, (p1[1] - p0[1]) / length)
        centre = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
        depth = spec.wall_thickness + 2.0 * CASING_DEPTH
        for side, sign in (("a", -1), ("b", 1)):
            offset = sign * (length / 2.0 + CASING_WIDTH / 2.0)
            centre_i = (centre[0] + direction[0] * offset, centre[1] + direction[1] * offset)
            objects.append(
                shell_mod.add_oriented_box(
                    f"window_casing_{index}_{side}", centre_i, direction, CASING_WIDTH, depth,
                    window.sill, window.sill + window.height, collection,
                )
            )
        objects.append(
            shell_mod.add_oriented_box(
                f"window_casing_{index}_head", centre, direction, length + 2 * CASING_WIDTH, depth,
                window.sill + window.height, window.sill + window.height + CASING_WIDTH, collection,
            )
        )
        # sill board, protruding into the room on the inner face
        inward = spec_mod.inward_normal(window.room, centre, spec)
        sill_centre = (
            centre[0] + inward[0] * (spec.wall_thickness / 2.0 + SILL_OVERHANG / 2.0),
            centre[1] + inward[1] * (spec.wall_thickness / 2.0 + SILL_OVERHANG / 2.0),
        )
        objects.append(
            shell_mod.add_oriented_box(
                f"window_sill_{index}", sill_centre, direction, length + 2 * CASING_WIDTH,
                spec.wall_thickness + SILL_OVERHANG, window.sill - SILL_THICKNESS, window.sill, collection,
            )
        )
        objects.append(
            shell_mod.add_oriented_box(
                f"window_casing_{index}_bottom", centre, direction, length + 2 * CASING_WIDTH, depth,
                window.sill - SILL_THICKNESS - 0.02, window.sill - SILL_THICKNESS, collection,
            )
        )
    return objects


def build_glass(spec: spec_mod.SceneSpec, collection) -> List:
    """One thin pane per window, centred in the wall."""
    objects = []
    for index, window in enumerate(spec.windows):
        p0 = (float(window.wall[0][0]), float(window.wall[0][1]))
        p1 = (float(window.wall[1][0]), float(window.wall[1][1]))
        length = math.dist(p0, p1)
        direction = ((p1[0] - p0[0]) / length, (p1[1] - p0[1]) / length)
        centre = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
        objects.append(
            shell_mod.add_oriented_box(
                f"window_glass_{index}", centre, direction, length, GLASS_THICKNESS,
                window.sill, window.sill + window.height, collection,
            )
        )
    return objects


def make_glass_material(name: str = "mat_window_glass"):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.02
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["IOR"].default_value = 1.45
    return material


def bevel_shell(objects: Sequence, width: float = BEVEL_WIDTH, segments: int = BEVEL_SEGMENTS) -> int:
    """Add (not apply) a small bevel modifier to every shell object."""
    count = 0
    for obj in objects:
        modifier = obj.modifiers.new(name="bevel", type="BEVEL")
        modifier.width = width
        modifier.segments = segments
        modifier.limit_method = "ANGLE"
        modifier.angle_limit = math.radians(35.0)
        modifier.harden_normals = False
        count += 1
    return count


def make_rug(center, size, rotation_deg: float, material, collection, name: str = "rug"):
    """A thin rounded-ish rug slab lying on the floor."""
    length, width = float(size[0]), float(size[1])
    return shell_mod.add_oriented_box(
        name=name,
        center_xy=center,
        direction=(math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))),
        length=length,
        depth=width,
        z_bottom=0.0,
        z_top=0.012,
        collection=collection,
    )


def make_table_lamp(center, height: float = 0.34, radius: float = 0.11, collection=None,
                    name: str = "table_lamp") -> List:
    """A minimal but decent bedside lamp: tapered base + conical shade."""
    body = shell_mod.add_prism(
        f"{name}_base", _circle(center, radius * 0.32), 0.0, height * 0.42, collection
    )
    stem = shell_mod.add_prism(
        f"{name}_stem", _circle(center, radius * 0.09), height * 0.42, height * 0.58, collection
    )
    shade = _cone(f"{name}_shade", center, radius * 0.55, radius, height * 0.58, height, collection)
    return [body, stem, shade]


def _circle(center, radius: float, segments: int = 20):
    return [
        (center[0] + radius * math.cos(2 * math.pi * i / segments),
         center[1] + radius * math.sin(2 * math.pi * i / segments))
        for i in range(segments)
    ]


def _cone(name, center, r_bottom: float, r_top: float, z0: float, z1: float, collection):
    """Open-ended truncated cone (shade) built from explicit mesh data."""
    segments = 24
    vertices = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        vertices.append((center[0] + r_bottom * math.cos(angle), center[1] + r_bottom * math.sin(angle), z0))
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        vertices.append((center[0] + r_top * math.cos(angle), center[1] + r_top * math.sin(angle), z1))
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, j + segments, i + segments])
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.validate()
    return shell_mod.link_object(name, mesh, collection)
