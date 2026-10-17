"""Placement and circulation checks for ``blender_agent`` scene specs.

Pure stdlib (no ``bpy`` / ``numpy``) so it runs on any host python and is
imported by ``blender_agent/layout_gen.py``.  The numbers below are the layout
contract shared with the generator:

* every item must resolve to a bounding box: ``ph:<id>`` from
  ``third_party/assets/polyhaven/index.json`` (``api_dimensions_mm`` are
  millimetres), ``builtin:rug`` from its ``size``, ``builtin:table_lamp`` from
  the fixed prop size.  A yaw that is not a multiple of 90 still yields a
  conservative AABB (the AABB of the rotated box), so the generator keeps
  rotations on the 90 degree grid.  ``scale`` is allowed for every scheme and
  multiplies the whole resolved size: the builder scales the same geometry by
  the same factor (``fixtures.world_boxes`` for the ``builtin:`` parts), so the
  checked box *is* the built one.  ``z`` is a world height and
  ``mount: "ceiling"`` hangs the item from its own room's ceiling -- both are
  resolved by :func:`item_world_z`, the one rule the builder calls too, so a
  non-zero storey elevation is added exactly once.
* ``wall_gap`` is measured from the footprint to the room boundary *line*, which
  is the wall centre line: rooms abut on centre lines and a 0.12 m wall reaches
  0.06 m into each of them.  Every item needs ``wall_gap >= 0.05`` (nothing
  embedded in a wall).  Items are then either *wall hugging*
  (``0.05 <= gap <= 0.35``) or *free standing* (``gap >= 0.60``, the
  circulation rule); floor-flat (rug), ceiling and wall mounted items only need
  to sit inside the room.  An explicit ``"placement"`` field in the spec wins
  over the geometric inference, so a hand written spec can state its intent.
* collisions are 3D AABB overlaps with the intended exemptions: rugs, ceiling
  mounted items, wall art that starts above the item it hangs over, items
  marked ``attached_to`` another item, and items standing on another item's top
  face (a lamp on a nightstand).
* doors stay clear: no blocking furniture within 0.15 m of the door approach
  rectangle (door width + 0.4 m, 0.85 m deep on each side of the wall).  The
  zone only guards the doorway itself; keeping the whole passage walkable is the
  circulation check's job.  Doors and their approach zones belong to one storey:
  furniture on another floor at the same XY never blocks them.
* circulation is 2.5D: every storey gets its own 5 cm walkable grid.  A cell is
  *free* when it is inside a room of that storey inset by the wall half
  thickness or inside a door corridor of that storey, and not inside a blocking
  footprint dilated by 0.20 m -- a blocking item intersects its own storey's
  walkable band (floor + 0.20 m .. floor + 1.60 m), so rugs, wall art and
  ceiling lamps never block and furniture one storey up never blocks the floor
  below.  Slab openings *and stair footprints* are holes (not walkable) -- the
  floor of a storey is only where a real slab is.  BFS from the centre of the
  first opening must reach every room (at least 25% of that room's walkable
  cells, never fewer than 8), every camera and -- through the ``stair`` /
  ``ramp`` connectors -- every storey.  A stair is not floor: it is a walk track
  of nodes that carry their own height (``spec.stair_track`` / ``ramp_track``),
  and only its entry node (lower floor) and exit node (upper floor) join a
  storey grid.  Two storeys stack in XY without either being false-blocked.
* cameras sit inside their room and at least 0.50 m away from blocking
  furniture of their own storey, and they *stand* on a real surface: the storey
  floor slab has to be under the camera (not a slab opening, a void or a stair
  tread) with ``CAMERA_EYE_HEIGHT_M`` (+- tolerance) between the surface and the
  camera -- the rule that rejects a stand above a stairwell.
* stairs and ramps are accepted on their own numbers as well: riser / tread /
  width / pitch / landing for both the straight run and the switchback, the
  fly-through opening in the slab above, the walk-through headroom (>= 2.0 m
  measured against the slab) and a clear stair mouth -- no blocking furniture on
  the stair, in front of its first riser or in front of its top step.  The walk
  track is measured too: step rise, pitch, monotonic climb and the body clearance
  above every node along the way.
* a lower room's ceiling is only skipped where an upper storey's slab really
  replaces it (same plane *and* covering the area); the rest keeps its own
  ceiling slab, and ``slab_openings`` stay open as declared voids.  Every
  obstruction test (headroom, walk track) uses the same per-area rule.
* an item has to fit under the body that is really above it -- the 3D
  containment rule (``furniture_containment``).  Its footprint is sampled and
  measured against the *lowest* slab underside / owned ceiling over those
  points, never against one height per storey: where an upper storey covers the
  room the binding surface is that slab's underside, where the room keeps its
  own ceiling it is the ceiling plane, and where a void or a double-height
  space is declared the item may rise to the first real slab above it.  A base
  may not sink into its own storey's slab, a floor-standing item may not stand
  over a slab opening of its own storey (nothing under it), and a ``mount:
  "ceiling"`` item has to reach the ceiling it claims to hang from -- a pendant
  over a void sees a slab far above and fails, because there is nothing to
  attach to.  Items on different storeys at the same XY are never measured
  against each other.
* the same rule runs again on the geometry as built: ``check_final_geometry``
  and the CLI ``--final-bboxes`` / ``--final-manifest`` take the post-settle
  world boxes and re-run every box-based rule on them, reporting what settle /
  repair moved, so a built scene can be re-checked without Blender.

CLI::

    python blender_agent/checks.py spec.json [--assets-index P] [--report R]
    python blender_agent/checks.py spec.json --final-bboxes boxes.json [--report R]
    python blender_agent/checks.py --final-manifest build/build_manifest.json [--report R]

prints a PASS/FAIL line per check and exits non-zero when any check fails.  The
``--final-*`` forms do not check the authored spec: they check the boxes that
were really built (``{"name", "room", "bbox": [x0, y0, z0, x1, y1, z1]}``, or a
build manifest whose furniture entries carry ``bbox_z`` from settle).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import fixtures as fixtures_mod  # noqa: E402
from blender_agent import spec as spec_mod  # noqa: E402

_MERGED_INDEX = os.path.join(REPO_ROOT, "third_party", "assets", "merged_index.json")
_POLYHAVEN_INDEX = os.path.join(REPO_ROOT, "third_party", "assets", "polyhaven", "index.json")
DEFAULT_ASSETS_INDEX = _MERGED_INDEX if os.path.isfile(_MERGED_INDEX) else _POLYHAVEN_INDEX

WALK_BAND_M: Tuple[float, float] = (0.20, 1.60)
WALL_GAP_MIN_M = 0.05
WALL_GAP_HUG_MAX_M = 0.35
WALL_GAP_FREE_MIN_M = 0.60
FURNITURE_DILATION_M = 0.20
DOOR_CLEARANCE_M = 0.15
DOOR_ZONE_DEPTH_M = 0.85
DOOR_ZONE_EXTRA_W_M = 0.40
CAMERA_FURNITURE_GAP_M = 0.50
CAMERA_WALL_GAP_M = 0.10
GRID_M = 0.05
ROOM_REACH_MIN_CELLS = 8
ROOM_REACH_MIN_FRACTION = 0.25

# 3D containment (furniture against the real slabs / ceilings)
CEILING_TOUCH_TOL_M = 0.005      # a top face may reach a ceiling plane this far
CEILING_HUG_MAX_M = 0.10         # a ceiling mount has to reach the body it hangs from
SLAB_SINK_TOL_M = 0.005          # a base may sink this far into its own storey's slab
CONTAINMENT_GRID_M = 0.20        # footprint sampling step of the per-area rule
CONTAINMENT_MAX_INTERVALS = 20   # ... at most this many intervals per axis
CONTAINMENT_VOID_SHARE = 0.5     # share of the footprint that may hang over an opening
FLAT_FINISH_MAX_M = 0.05         # a piece this thin is a floor finish, not a body

# generated props (blender_agent/details.py) that have no Poly Haven entry
BUILTIN_DIMS_M: Dict[str, Tuple[float, float, float]] = {
    "builtin:table_lamp": (0.22, 0.22, 0.34),
}
# functional furniture (blender_agent/fixtures.py: kitchen run, bathroom set,
# desk ...) registers its measured parts the same way, so a ``builtin:`` item
# sizes exactly like the geometry the builder emits
BUILTIN_DIMS_M.update(fixtures_mod.BUILTIN_DIMS_M)
RUG_THICKNESS_M = 0.04

FLOOR_FLAT, WALL_HUG, FREE_STAND, CEILING, MOUNTED = (
    "floor_flat", "wall_hug", "free_stand", "ceiling", "mounted",
)
PLACEMENT_CLASSES = (FLOOR_FLAT, WALL_HUG, FREE_STAND, CEILING, MOUNTED)


# --------------------------------------------------------------------------- #
# assets
# --------------------------------------------------------------------------- #
def load_assets_index(path: str = DEFAULT_ASSETS_INDEX) -> Dict[str, dict]:
    """``{asset id: {"dims_m": (w, d, h), "categories": [...], "ref": ..., ...}}``.

    Accepts the Poly Haven cache shape (``{"models": {...}}``) and the flat
    merged registry from ``tools/make_merged_assets.py``; ``ref`` is passed
    through so the layout generator can emit source-aware asset references.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    models = raw.get("models") if isinstance(raw.get("models"), dict) else raw
    out: Dict[str, dict] = {}
    for asset_id, entry in (models or {}).items():
        if not isinstance(entry, dict):
            continue
        dims_mm = entry.get("api_dimensions_mm")
        dims_m = None
        if dims_mm and len(dims_mm) == 3 and all(float(v) > 0.0 for v in dims_mm):
            dims_m = tuple(float(v) / 1000.0 for v in dims_mm)
        out[asset_id] = {
            "dims_m": dims_m,
            "categories": [str(c) for c in (entry.get("categories") or [])],
            "attributes": entry.get("attributes") or {},
            "ref": entry.get("ref"),
            "source": entry.get("source"),
        }
    return out


def item_dims(asset: str, index: Dict[str, dict], size: Optional[Sequence[float]] = None,
              scale: float = 1.0) -> Optional[Tuple[float, float, float]]:
    """Un-rotated ``(width_x, depth_y, height_z)`` of a furniture item in metres.

    ``scale`` (``FurnitureSpec.scale``) multiplies the resolved size for every
    scheme -- a ``builtin:`` part is scaled from the measured table
    (``fixtures.BUILTIN_DIMS_M``), whose boxes the builder scales with the same
    factor, and a rug from its ``size``; its thickness stays the fixed prop
    thickness.
    """
    scheme, _, asset_id = asset.partition(":")
    if scheme == "builtin":
        if asset_id == "rug":
            if not size or len(size) < 2:
                return None
            return (float(size[0]) * scale, float(size[1]) * scale, RUG_THICKNESS_M)
        dims = BUILTIN_DIMS_M.get(asset)
        return tuple(v * scale for v in dims) if dims else None
    entry = index.get(asset_id)
    if entry is None or entry.get("dims_m") is None:
        return None
    return tuple(v * scale for v in entry["dims_m"])


def rotated_aabb(pos: Sequence[float], dims: Sequence[float], rot_z_deg: float):
    """XY AABB of a ``dims``-sized box centred on ``pos`` and yawed by ``rot_z_deg``."""
    angle = math.radians(float(rot_z_deg))
    cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
    half_x = 0.5 * (dims[0] * cos_a + dims[1] * sin_a)
    half_y = 0.5 * (dims[0] * sin_a + dims[1] * cos_a)
    return (float(pos[0]) - half_x, float(pos[1]) - half_y,
            float(pos[0]) + half_x, float(pos[1]) + half_y)


@dataclass
class ItemBox:
    """One furniture item reduced to a 3D box plus its placement class."""

    name: str
    asset: str
    room: str
    aabb: Tuple[float, float, float, float]
    z: Tuple[float, float]
    dims: Optional[Tuple[float, float, float]]
    placement: str
    rot_z_deg: float
    rotation_on_grid: bool
    floor_z: float = 0.0
    floor_id: str = ""
    attached_to: Optional[str] = None
    mount: str = ""  # "floor" / "ceiling" / "wall" as authored ("" when inferred)

    @property
    def ceiling_mounted(self) -> bool:
        """Does this item claim to hang from the ceiling (``mount: "ceiling"``)?"""
        return self.placement == CEILING or self.mount == "ceiling"

    @property
    def walk_band(self) -> Tuple[float, float]:
        """Walkable Z band of this item's storey (legacy/ground floor: 0.20..1.60)."""
        return (self.floor_z + WALK_BAND_M[0], self.floor_z + WALK_BAND_M[1])

    @property
    def blocks_walking(self) -> bool:
        band = self.walk_band
        return self.placement != FLOOR_FLAT and self.z[0] < band[1] and self.z[1] > band[0]

    @property
    def blocks_camera(self) -> bool:
        band = self.walk_band
        return self.placement not in (FLOOR_FLAT, CEILING, MOUNTED) and self.z[0] < band[1] \
            and self.z[1] > band[0]



def item_world_z(spec: "spec_mod.SceneSpec", item: "spec_mod.FurnitureSpec",
                 dims: Sequence[float]) -> Tuple[float, float, float]:
    """``(floor_z, bottom_z, top_z)`` of one item in world coordinates.

    The single local-to-world height rule of the pipeline -- ``checks.py`` and
    ``run_build.py`` both call it, so the checked box and the built geometry
    cannot drift apart:

    * ``item.z`` is a *world* height (the ``run_build`` contract); without one
      the item stands on the floor of its room, at its storey's elevation.
    * ``mount == "ceiling"`` hangs the item from the ceiling of its own room:
      its top is the room top and its bottom one object height below, which is a
      *local* height and must never be offset by the storey's elevation again.

    A non-zero storey elevation is therefore added exactly once, whichever
    branch runs.
    """
    floor_z = float(spec.room_elevation(item.room))
    height = float(dims[2])
    if getattr(item, "mount", "") == "ceiling":
        bottom = spec.room_top(item.room) - height
    elif item.z is not None:
        bottom = float(item.z)
    else:
        bottom = floor_z
    return floor_z, bottom, bottom + height


def item_boxes(spec: "spec_mod.SceneSpec", index: Dict[str, dict]) -> Tuple[List[ItemBox], List[str]]:
    """Build boxes for every furniture item; the second value lists unknown assets."""
    boxes: List[ItemBox] = []
    unknown: List[str] = []
    for i, item in enumerate(spec.furniture):
        dims = item_dims(item.asset, index, item.size, item.scale)
        name = item.name or f"{item.asset.split(':', 1)[-1]}_{i}"
        if dims is None:
            unknown.append(f"{name}: no measured size for '{item.asset}'")
            continue
        placement = str(getattr(item, "placement", "") or "")
        if placement not in PLACEMENT_CLASSES:
            placement = ""
        floor = spec.room_floor(item.room)
        floor_z, bottom_z, top_z = item_world_z(spec, item, dims)
        # a world z on the room's own floor is "standing on the floor"; anything
        # lifted off it (relative to *that* floor) is mounted or stacked
        local_z = bottom_z - floor_z
        if item.mount == "ceiling":
            placement = CEILING
        elif placement == "" and item.z is not None and local_z >= 0.5:
            placement = MOUNTED
        elif placement == "" and item.asset == "builtin:rug":
            placement = FLOOR_FLAT
        rot = float(item.rot_z_deg)
        boxes.append(
            ItemBox(
                name=name,
                asset=item.asset,
                room=item.room,
                aabb=rotated_aabb(item.pos, dims, rot),
                z=(bottom_z, top_z),
                dims=dims,
                placement=placement,  # filled below when empty
                rot_z_deg=rot,
                rotation_on_grid=abs(rot / 90.0 - round(rot / 90.0)) < 1e-9,
                floor_z=floor_z,
                floor_id=floor.id,
                attached_to=(str(getattr(item, "attached_to", "")) or None),
                mount=str(getattr(item, "mount", "") or ""),
            )
        )
    return boxes, unknown


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def polygon_edges(polygon: Sequence[Sequence[float]]):
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        yield (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))


def inset_polygon(polygon: Sequence[Sequence[float]], distance: float) -> List[List[float]]:
    """Polygon shrunk by ``distance`` (edge offset + line intersection).

    Exact for the axis-aligned rectangles the generator emits and safe for
    convex polygons; a degenerate result means the room is too small.
    """
    edges = list(polygon_edges(polygon))
    lines = []
    for (ax, ay), (bx, by) in edges:
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux  # inward normal for counter-clockwise polygons
        lines.append((ax + nx * distance, ay + ny * distance, ux, uy))
    out: List[List[float]] = []
    count = len(lines)
    for i in range(count):
        x1, y1, u1x, u1y = lines[i - 1]
        x2, y2, u2x, u2y = lines[i]
        det = u1x * u2y - u1y * u2x
        if abs(det) < 1e-9:
            out.append([x2, y2])
            continue
        t = ((x2 - x1) * u2y - (y2 - y1) * u2x) / det
        out.append([x1 + u1x * t, y1 + u1y * t])
    return out


def point_in_polygon(point, polygon) -> bool:
    return spec_mod.point_in_polygon(point, polygon)


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def rect_segment_distance(rect, ax: float, ay: float, bx: float, by: float) -> float:
    """Smallest distance between an axis-aligned rectangle and a segment (0 when they meet)."""
    x0, y0, x1, y1 = rect
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    lo_x, hi_x = min(ax, bx), max(ax, bx)
    lo_y, hi_y = min(ay, by), max(ay, by)
    if x1 >= lo_x and x0 <= hi_x and y1 >= lo_y and y0 <= hi_y:
        # bounding boxes overlap: the segment may still pass beside the rectangle,
        # so test the four rectangle edges against it explicitly
        for (cx, cy), (dx_, dy_) in zip(corners, corners[1:] + corners[:1]):
            if _segments_intersect(ax, ay, bx, by, cx, cy, dx_, dy_):
                return 0.0
    return min(_point_segment_distance(cx, cy, ax, ay, bx, by) for cx, cy in corners)


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def wall_gap(rect, polygon) -> float:
    """Distance from a footprint to the room boundary line (negative when outside)."""
    distance = min(rect_segment_distance(rect, ax, ay, bx, by)
                   for (ax, ay), (bx, by) in polygon_edges(polygon))
    for cx, cy in ((rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])):
        if not point_in_polygon((cx, cy), polygon):
            return -distance
    return distance


def rect_distance(a, b) -> float:
    """Distance between two axis-aligned rectangles (0 when they overlap)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def boxes_overlap3d(a: ItemBox, b: ItemBox, tol: float = 1e-3) -> bool:
    if a.z[0] > b.z[1] - tol or b.z[0] > a.z[1] - tol:
        return False
    return not (a.aabb[2] <= b.aabb[0] + tol or b.aabb[2] <= a.aabb[0] + tol
                or a.aabb[3] <= b.aabb[1] + tol or b.aabb[3] <= a.aabb[1] + tol)


def collision_exempt(a: ItemBox, b: ItemBox) -> Optional[str]:
    """Why an overlapping pair is intentional, or ``None`` when it is a real clash."""
    if FLOOR_FLAT in (a.placement, b.placement):
        return "floor_flat"  # a rug lives under other furniture
    if CEILING in (a.placement, b.placement):
        return "ceiling_mounted"
    for first, second in ((a, b), (b, a)):
        if second.attached_to and second.attached_to == first.name:
            return "supported"  # the spec says it rides on the other item
        if second.z[0] >= first.z[1] - 0.03 and \
                second.aabb[0] >= first.aabb[0] - 0.05 and second.aabb[2] <= first.aabb[2] + 0.05 and \
                second.aabb[1] >= first.aabb[1] - 0.05 and second.aabb[3] <= first.aabb[3] + 0.05:
            return "stacked"  # a lamp standing on a table top
    return None


def infer_placement(box: ItemBox, room, wall_gap_m: float) -> str:
    if box.placement:
        return box.placement
    if box.asset == "builtin:rug":
        return FLOOR_FLAT
    if box.z[0] - box.floor_z >= 1.2:
        return MOUNTED
    if wall_gap_m <= WALL_GAP_HUG_MAX_M:
        return WALL_HUG
    return FREE_STAND


# --------------------------------------------------------------------------- #
# room / door helpers
# --------------------------------------------------------------------------- #
@dataclass
class DoorZone:
    a: str
    b: str
    center: Tuple[float, float]
    width: float
    height: float
    direction: Tuple[float, float]
    rects: Dict[str, Tuple[float, float, float, float]]
    floor_id: str = ""
    floor_z: float = 0.0


def door_zones(spec: "spec_mod.SceneSpec", wall_thickness: float) -> List[DoorZone]:
    """Door approach rectangles, one per adjoining room (per storey)."""
    segments = spec.wall_segments()
    zones: List[DoorZone] = []
    for connector in spec.door_connections():
        center = connector.center
        if center is None:
            continue
        floor = spec.room_floor(connector.from_room)
        segment = spec_mod.wall_segment_for_point(
            segments, center, connector.from_room,
            floor_id=connector.from_floor if spec.floors else None,
        )
        if segment is None:
            continue
        ux, uy = segment.direction
        half_w = connector.width / 2.0 + DOOR_ZONE_EXTRA_W_M / 2.0
        half_d = wall_thickness / 2.0 + DOOR_ZONE_DEPTH_M
        rects: Dict[str, Tuple[float, float, float, float]] = {}
        for room_id in (connector.from_room, connector.to_room):
            nx, ny = spec_mod.inward_normal(room_id, center, spec)
            ax, ay = center[0] + nx * half_d, center[1] + ny * half_d
            bx, by = center[0] - nx * wall_thickness / 4.0, center[1] - ny * wall_thickness / 4.0
            rects[room_id] = (min(ax, bx) - abs(ux) * half_w, min(ay, by) - abs(uy) * half_w,
                              max(ax, bx) + abs(ux) * half_w, max(ay, by) + abs(uy) * half_w)
        zones.append(DoorZone(a=connector.from_room, b=connector.to_room,
                              center=(float(center[0]), float(center[1])),
                              width=float(connector.width),
                              height=float(connector.height or spec_mod.DEFAULT_DOOR_HEIGHT_M),
                              direction=(ux, uy), rects=rects,
                              floor_id=floor.id, floor_z=float(floor.elevation)))
    return zones


def corridor_rect(zone: DoorZone, wall_thickness: float) -> Tuple[float, float, float, float]:
    """Grid cells inside this rectangle bridge the two rooms at the door."""
    ux, uy = zone.direction
    half_w = zone.width / 2.0 - 0.03
    half_d = wall_thickness / 2.0 + 0.10
    cx, cy = zone.center
    return (cx - abs(ux) * half_w - abs(uy) * half_d, cy - abs(uy) * half_w - abs(ux) * half_d,
            cx + abs(ux) * half_w + abs(uy) * half_d, cy + abs(uy) * half_w + abs(ux) * half_d)


# --------------------------------------------------------------------------- #
# 3D containment: what the real slabs and ceilings leave for an item
# --------------------------------------------------------------------------- #
@dataclass
class OverheadFace:
    """One real horizontal body above a point: a slab underside or a ceiling."""

    bottom: float
    kind: str        # "slab" | "ceiling"
    floor: str
    room: str
    label: str


def ceiling_parts_index(spec: "spec_mod.SceneSpec") -> Dict[str, List[List[List[float]]]]:
    """``{room id: ceiling outlines the room still owns}``, clipped once per room.

    :func:`overhead_faces` tests points against this instead of calling
    ``spec.ceiling_above`` per sample; the polygon difference behind it is the
    same one (:func:`room_ceiling_parts`), just not recomputed per sample.
    """
    return {room.id: spec.ceiling_regions(room.id) for room in spec.rooms}


def overhead_faces(spec: "spec_mod.SceneSpec", point: Sequence[float],
                   ceilings: Optional[Dict[str, List[List[List[float]]]]] = None
                   ) -> List[OverheadFace]:
    """Every slab underside and owned ceiling above ``point``, across all storeys.

    The per-area rule the stair headroom and the walk track already use: a
    storey's slab counts where a room of that storey covers the point and the
    point is not inside one of that storey's slab openings; a room's ceiling
    counts only where the room still owns it, i.e. where no upper slab replaces
    it.  A declared void therefore contributes neither, so the first body above
    a void is the slab (or the ceiling) of a storey further up.
    """
    parts = ceilings if ceilings is not None else ceiling_parts_index(spec)
    storeys = spec.floors or [_synthetic_check_floor(spec)]
    out: List[OverheadFace] = []
    for floor in storeys:
        rooms = spec.floor_rooms(floor.id) if spec.floors else list(spec.rooms)
        name = floor.id if spec.floors else "f0"
        openings = spec.slab_openings_for(floor.id) if spec.floors else []
        opened = any(point_in_polygon(point, opening.outline) for opening in openings)
        for room in rooms:
            if not point_in_polygon(point, room.polygon):
                continue
            if not opened:
                out.append(OverheadFace(bottom=floor.elevation - floor.slab_thickness, kind="slab",
                                        floor=name, room=room.id,
                                        label=f"slab of floor '{name}'"))
            ceiling = floor.elevation + spec.room_height(room.id)
            if any(point_in_polygon(point, part) for part in parts.get(room.id, ())):
                out.append(OverheadFace(bottom=ceiling, kind="ceiling", floor=name, room=room.id,
                                        label=f"ceiling of room '{room.id}'"))
    return out


def _axis_samples(lo: float, hi: float, grid_m: float) -> List[float]:
    intervals = max(1, min(CONTAINMENT_MAX_INTERVALS, int(math.ceil((hi - lo) / grid_m))))
    span = hi - lo
    values = [lo + span * i / intervals for i in range(intervals + 1)]
    middle = 0.5 * (lo + hi)
    if all(abs(value - middle) > 1e-9 for value in values):
        values.append(middle)
    return sorted(values)


def footprint_samples(aabb: Sequence[float], grid_m: float = CONTAINMENT_GRID_M
                      ) -> List[Tuple[float, float]]:
    """XY points an item is measured at: its corners, its centre and a lattice.

    The body above an item is not one number per room, so the item is probed
    where its footprint really is -- a cabinet that straddles the edge of a
    partial upper floor is bounded by the lower of the two surfaces it meets.
    """
    xs = _axis_samples(float(aabb[0]), float(aabb[2]), grid_m)
    ys = _axis_samples(float(aabb[1]), float(aabb[3]), grid_m)
    return [(x, y) for x in xs for y in ys]


def containment_report(spec: "spec_mod.SceneSpec", boxes: Sequence[ItemBox],
                       ceilings: Optional[Dict[str, List[List[List[float]]]]] = None) -> dict:
    """Measure every item against the real body above and below it.

    Per item: the lowest slab underside / owned ceiling over its footprint is
    what it has to fit under (a top face may touch that plane, not pass it), a
    base may not sink into its own storey's slab, a floor-standing item may not
    stand over a slab opening of its own storey, and a ``mount: "ceiling"``
    item must reach the body it hangs from -- a pendant over a declared void
    sees the slab of an upper storey metres above and fails, because nothing is
    there to attach it to.

    Returns ``{"items": [...], "violations": [...]}``; the per-item entries
    carry the measured surface, the clearance to it and the share of the
    footprint that hangs over an opening, which is what the final-geometry
    re-check reports as well.
    """
    parts = ceilings if ceilings is not None else ceiling_parts_index(spec)
    items: List[dict] = []
    violations: List[str] = []
    for box in boxes:
        base, top = float(box.z[0]), float(box.z[1])
        floor = spec.room_floor(box.room)
        floor_name = floor.id if spec.floors else "f0"
        samples = footprint_samples(box.aabb)
        lowest: Optional[OverheadFace] = None
        opened = 0
        own_ceiling = 0
        parts_of_room = parts.get(box.room, ())
        for point in samples:
            if spec.floors and spec.in_slab_opening(floor.id, point):
                opened += 1
            if any(point_in_polygon(point, part) for part in parts_of_room):
                own_ceiling += 1
            for face in overhead_faces(spec, point, parts):
                # a face at or below the item's own floor plane belongs to the sink
                # rule below, not to "pokes through the ceiling above"
                if face.bottom < max(base, floor.elevation) - CEILING_TOUCH_TOL_M:
                    continue
                if lowest is None or face.bottom < lowest.bottom - 1e-9:
                    lowest = face
        entry = {
            "name": box.name,
            "room": box.room, "floor": floor_name, "mount": box.mount or "floor",
            "placement": box.placement, "z": [round(base, 3), round(top, 3)],
            "samples": len(samples),
            "over_opening_share": round(opened / len(samples), 3) if samples else 0.0,
            "own_ceiling_share": round(own_ceiling / len(samples), 3) if samples else 0.0,
            "lowest_face": None if lowest is None else {
                "bottom": round(lowest.bottom, 3), "kind": lowest.kind, "floor": lowest.floor,
                "room": lowest.room, "label": lowest.label},
            "clearance_m": None if lowest is None else round(lowest.bottom - top, 3),
        }
        items.append(entry)

        if base < floor.elevation - SLAB_SINK_TOL_M and \
                box.placement != FLOOR_FLAT and top - base > FLAT_FINISH_MAX_M:
            # a floor finish (a rug, a sheet) lies *on* the slab and settle pushes
            # it into the floor plane by design; a body may not sink into it
            violations.append(
                f"{box.name} ({box.room}): base at {base:.3f} m sinks "
                f"{floor.elevation - base:.3f} m into the slab of floor '{floor_name}'"
            )
        if lowest is None:
            if box.ceiling_mounted:
                violations.append(
                    f"{box.name} ({box.room}): mounted on the ceiling at {top:.3f} m but nothing is "
                    f"above its footprint (no slab and no ceiling) -- there is nothing to hang from"
                )
        elif top > lowest.bottom + CEILING_TOUCH_TOL_M:
            violations.append(
                f"{box.name} ({box.room}): top at {top:.3f} m pokes "
                f"{top - lowest.bottom:.3f} m through the {lowest.label} at "
                f"{lowest.bottom:.3f} m"
            )
        if box.ceiling_mounted:
            if lowest is not None and lowest.bottom - top > CEILING_HUG_MAX_M + 1e-9:
                where = ("its own room's ceiling is not there over the footprint: a declared void "
                         "or a double-height space"
                         if entry["own_ceiling_share"] < 0.5 else
                         "no slab and no ceiling sits over its footprint")
                violations.append(
                    f"{box.name} ({box.room}): mounted on the ceiling but its top at {top:.3f} m is "
                    f"{lowest.bottom - top:.3f} m below the {lowest.label} at {lowest.bottom:.3f} m "
                    f"({where}) -- a ceiling-mounted item has to reach the ceiling of its own room "
                    f"(max {CEILING_HUG_MAX_M} m below it)"
                )
        elif entry["over_opening_share"] >= CONTAINMENT_VOID_SHARE \
                and abs(base - floor.elevation) <= SLAB_SINK_TOL_M + 1e-9:
            violations.append(
                f"{box.name} ({box.room}): stands on the floor but "
                f"{entry['over_opening_share'] * 100:.0f}% of its footprint is over a slab opening "
                f"of floor '{floor_name}' -- nothing supports it there"
            )
    return {"items": items, "violations": violations}


def _item_box(entry: dict, spec: Optional["spec_mod.SceneSpec"] = None,
              rooms_by_name: Optional[Dict[str, str]] = None) -> Tuple[Optional[ItemBox], str]:
    """One world box from a dict: ``bbox`` (x0,y0,z0,x1,y1,z1) or ``aabb`` + ``z``."""
    room = str(entry.get("room") or "")
    name = str(entry.get("name") or "")
    if not room and rooms_by_name:
        room = rooms_by_name.get(name, "")
    if not room:
        return None, f"{name or '?'}: no room (needed to find its slabs and ceilings)"
    bbox = entry.get("bbox")
    if bbox is not None:
        if len(bbox) != 6:
            return None, f"{name}: 'bbox' must be [x0, y0, z0, x1, y1, z1]"
        x0, y0, z0, x1, y1, z1 = (float(v) for v in bbox)
        aabb, z = (x0, y0, x1, y1), (z0, z1)
    elif entry.get("aabb") is not None and entry.get("z") is not None:
        x0, y0, x1, y1 = (float(v) for v in entry["aabb"])
        z0, z1 = (float(v) for v in entry["z"])
        aabb, z = (x0, y0, x1, y1), (z0, z1)
    else:
        return None, f"{name}: needs a 'bbox' or 'aabb' + 'z'"
    mount = str(entry.get("mount") or "")
    placement = str(entry.get("placement") or "")
    if placement not in PLACEMENT_CLASSES:
        placement = CEILING if mount == "ceiling" else ""
    floor_z = 0.0
    floor_id = ""
    if spec is not None and room in {item.id for item in spec.rooms}:
        floor = spec.room_floor(room)
        floor_z, floor_id = float(floor.elevation), floor.id
    return ItemBox(name=name or f"item_{id(entry)}", asset=str(entry.get("asset") or ""),
                   room=room, aabb=aabb, z=z, dims=None, placement=placement,
                   rot_z_deg=float(entry.get("rot_z_deg") or 0.0), rotation_on_grid=True,
                   floor_z=floor_z, floor_id=floor_id, mount=mount), ""


def boxes_from_entries(entries: Sequence[dict], spec: Optional["spec_mod.SceneSpec"] = None
                       ) -> Tuple[List[ItemBox], List[str]]:
    """Item boxes from explicit world boxes; the second value lists unusable entries.

    The contract of the final-geometry re-check: whatever produced the boxes
    (settle, a repair pass, another tool) only has to provide the world AABB and
    the room, and every box-based rule can be measured on it.
    """
    rooms_by_name = {item.name: item.room for item in spec.furniture} if spec is not None else None
    boxes: List[ItemBox] = []
    notes: List[str] = []
    for entry in entries:
        box, note = _item_box(entry, spec, rooms_by_name)
        if box is None:
            notes.append(note)
        else:
            boxes.append(box)
    return boxes, notes


def _built_size_note(name: str, built: Sequence[float], dims: Sequence[float]) -> Optional[str]:
    """Note when the asset as built is not the size the registry measured.

    The builder records ``asset_object.dimensions * scale``; the checker sizes
    the same asset from its index entry.  A piece whose built body differs from
    the registry size (import rotation, local scaling inside the asset) is a
    drift between "checked" and "built" worth naming, so the final re-check
    reports it.  Sorted sizes, because the two may be expressed in different
    frames.
    """
    if dims is None:
        return None
    pairs = sorted(zip((float(v) for v in built), (float(v) for v in dims)),
                   key=lambda pair: pair[1])
    worst = max((abs(a - b), a, b) for a, b in pairs)
    if worst[0] <= 0.05:
        return None
    return (f"{name}: built size differs from the registry by up to {worst[0]:.3f} m "
            f"(built {worst[1]:.3f} m vs registry {worst[2]:.3f} m)")


def _manifest_boxes(manifest: dict, spec: "spec_mod.SceneSpec",
                    index: Optional[Dict[str, dict]] = None
                    ) -> Tuple[List[ItemBox], List[str]]:
    """Final boxes from a ``build_manifest.json``: settle's world Z on the placed footprint.

    The builder walks ``spec.furniture`` in order and records what it placed;
    settle rewrites ``bbox_z`` to the world Z range of the geometry it relaxed,
    which is the only part of the final pose the manifest carries.  So the box
    is the spec's own footprint (same ``rotated_aabb`` as the spec-time check)
    with the settled Z range -- the height really built, which is what the
    containment rule needs.  Lateral drift (millimetres, not recorded) is not
    re-checked here; a caller with true post-settle boxes uses the ``bbox``
    contract instead.
    """
    nominal, unknown = item_boxes(spec, index or {})
    notes: List[str] = list(unknown)
    boxes: List[ItemBox] = []
    cursor = 0
    for placement in manifest.get("furniture") or []:
        name = str(placement.get("name") or "")
        # entries follow the spec order, so the first not-yet-consumed item of the
        # same name is this one -- duplicate names ("bed" twice) stay distinct
        while cursor < len(nominal) and nominal[cursor].name != name:
            cursor += 1
        if cursor >= len(nominal):
            notes.append(f"{name or '?'}: no measured spec item of this name -- not re-checked")
            continue
        source = nominal[cursor]
        cursor += 1
        built = placement.get("size_m")
        if built and len(built) == 3:
            note = _built_size_note(name, built, source.dims)
            if note:
                notes.append(note)
        z_range = placement.get("bbox_z")
        if not (z_range and len(z_range) == 2 and float(z_range[1]) > float(z_range[0])):
            notes.append(f"{name}: no settled 'bbox_z' -- re-checked at its authored height")
            boxes.append(source)
            continue
        boxes.append(replace(source, z=(float(z_range[0]), float(z_range[1]))))
    return boxes, notes


def check_final_geometry(spec: "spec_mod.SceneSpec", boxes: Sequence[ItemBox],
                         assets_index: Optional[Dict[str, dict]] = None, grid_m: float = GRID_M,
                         authored: Optional[Sequence[ItemBox]] = None, source: str = "",
                         notes: Optional[Sequence[str]] = None) -> dict:
    """Re-run every box-based rule on the geometry as built (post-settle boxes).

    ``check_spec`` measures the spec; this measures what settle / repair left
    behind: through-wall, through-ceiling, collisions, door zones, stair
    clearance, circulation and containment all read the given boxes, and
    ``adjustments`` records what moved relative to the authored boxes, so a
    "repaired" scene cannot pass by being measured before the repair.  A run
    that measured no box at all fails ``final_geometry_measured`` when the spec
    has furniture: an empty re-check must never read as a pass.
    """
    report = check_spec(spec, assets_index, grid_m=grid_m, boxes=boxes)
    report["kind"] = "blender_agent_final_geometry"
    report["mode"] = "final_geometry"
    report["source"] = source
    measured = bool(boxes) or not spec.furniture
    report["checks"].append({"name": "final_geometry_measured", "pass": measured,
                             "boxes": len(boxes), "spec_items": len(spec.furniture),
                             "notes": list(notes or [])})
    if not measured:
        report["violations"].append(
            f"final geometry: the spec has {len(spec.furniture)} furniture items but the input "
            f"measured none -- the re-check proves nothing"
        )
        report["pass"] = False
    adjustments = []
    if authored is not None:
        by_name = {box.name: box for box in authored}
        for box in boxes:
            before = by_name.get(box.name)
            if before is None:
                adjustments.append({"name": box.name, "status": "not_in_spec"})
                continue
            entry = {
                "name": box.name, "room": box.room,
                "xy_shift_m": round(math.dist(
                    ((box.aabb[0] + box.aabb[2]) / 2.0, (box.aabb[1] + box.aabb[3]) / 2.0),
                    ((before.aabb[0] + before.aabb[2]) / 2.0,
                     (before.aabb[1] + before.aabb[3]) / 2.0)), 4),
                "z_before": [round(float(before.z[0]), 4), round(float(before.z[1]), 4)],
                "z_after": [round(float(box.z[0]), 4), round(float(box.z[1]), 4)],
                "z_shift_m": round(float(box.z[0]) - float(before.z[0]), 4),
                "height_shift_m": round((float(box.z[1]) - float(box.z[0]))
                                        - (float(before.z[1]) - float(before.z[0])), 4),
            }
            entry["changed"] = bool(entry["xy_shift_m"] > 1e-4 or abs(entry["z_shift_m"]) > 1e-4
                                    or abs(entry["height_shift_m"]) > 1e-4)
            adjustments.append(entry)
    report["final_geometry"] = {
        "boxes": {box.name: {"room": box.room, "placement": box.placement,
                             "aabb": [round(v, 3) for v in box.aabb],
                             "z": [round(v, 3) for v in box.z]} for box in boxes},
        "adjustments": adjustments,
        "moved": sum(1 for entry in adjustments if entry.get("changed")),
    }
    return report


def load_final_boxes(path: str, spec: Optional["spec_mod.SceneSpec"] = None,
                     index: Optional[Dict[str, dict]] = None
                     ) -> Tuple[Optional["spec_mod.SceneSpec"], List[ItemBox], List[str], str]:
    """Read the final-geometry input: a box list or a build manifest.

    Returns ``(spec, boxes, notes, source)``.  The first form is the explicit
    contract -- ``{"items": [{"name", "room", "bbox": [x0, y0, z0, x1, y1, z1]}]}``
    (a bare list works too); the second form is a ``build_manifest.json``, whose
    embedded spec is parsed and whose furniture entries carry settle's ``bbox_z``.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        data = {"items": data}
    if str(data.get("kind") or "") == "blender_agent_build":
        if spec is None:
            raw = data.get("spec")
            if not isinstance(raw, dict):
                raise spec_mod.SpecError(f"{path}: build manifest carries no 'spec' block")
            spec = spec_mod.parse_spec(raw, source_path=path)
        boxes, notes = _manifest_boxes(data, spec, index)
        return spec, boxes, notes, os.path.abspath(path)
    if spec is None:
        raise spec_mod.SpecError(f"{path}: a box list needs the spec that goes with it")
    boxes, notes = boxes_from_entries(data.get("items") or [], spec)
    return spec, boxes, notes, os.path.abspath(path)


# --------------------------------------------------------------------------- #
# the check itself
# --------------------------------------------------------------------------- #
def check_spec(spec: "spec_mod.SceneSpec", assets_index: Optional[Dict[str, dict]] = None,
               grid_m: float = GRID_M, boxes: Optional[Sequence[ItemBox]] = None) -> dict:
    """Run every placement rule. Returns a report dict with ``pass`` and ``checks``.

    ``boxes`` overrides the boxes resolved from the spec: the final-geometry
    re-check (:func:`check_final_geometry`) passes the post-settle world boxes
    so every box-based rule is measured on the geometry that was really built.
    """
    index = assets_index if assets_index is not None else load_assets_index()
    violations: List[str] = []
    checks: List[dict] = []

    def record(name: str, ok: bool, **payload) -> None:
        entry = {"name": name, "pass": bool(ok)}
        entry.update(payload)
        checks.append(entry)

    # -- sizes ------------------------------------------------------------- #
    resolved, unknown = item_boxes(spec, index)
    boxes = list(resolved) if boxes is None else list(boxes)
    record("assets_resolved", not unknown, unknown=unknown)
    violations.extend(unknown)

    # -- placement class + wall clearance ---------------------------------- #
    rooms = {room.id: room for room in spec.rooms}
    placements: Dict[str, str] = {}
    wall_gaps: Dict[str, float] = {}
    bad_gap: List[str] = []
    bad_inside: List[str] = []
    off_grid: List[str] = []
    for box in boxes:
        room = rooms[box.room]
        gap = wall_gap(box.aabb, room.polygon)
        wall_gaps[box.name] = round(gap, 4)
        placement = infer_placement(box, room, gap)
        placements[box.name] = placement
        box.placement = placement
        if not box.rotation_on_grid:
            off_grid.append(box.name)
        if gap < 0.0:
            bad_inside.append(f"{box.name}: footprint leaves room '{box.room}' (gap {gap:.3f} m)")
        elif gap < WALL_GAP_MIN_M - 1e-9:
            bad_gap.append(f"{box.name}: wall gap {gap:.3f} m < {WALL_GAP_MIN_M} m (embedded in a wall)")
        if placement == WALL_HUG and gap > WALL_GAP_HUG_MAX_M + 1e-9:
            bad_gap.append(f"{box.name}: wall hugging item sits {gap:.3f} m off the wall "
                           f"(max {WALL_GAP_HUG_MAX_M} m)")
        if placement == FREE_STAND and gap < WALL_GAP_FREE_MIN_M - 1e-9:
            bad_gap.append(f"{box.name}: free standing item only {gap:.3f} m from a wall "
                           f"(min {WALL_GAP_FREE_MIN_M} m)")
    record("furniture_inside_rooms", not bad_inside, violations=bad_inside,
           placements=placements, wall_gap_m=wall_gaps)
    record("furniture_wall_clearance", not bad_gap, violations=bad_gap)
    record("furniture_rotation_on_grid", not off_grid, violations=off_grid)
    violations.extend(bad_inside + bad_gap)

    # -- collisions -------------------------------------------------------- #
    clashes: List[str] = []
    exempt_pairs: List[dict] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if not boxes_overlap3d(a, b):
                continue
            reason = collision_exempt(a, b)
            if reason:
                exempt_pairs.append({"a": a.name, "b": b.name, "reason": reason})
            else:
                clashes.append(f"{a.name} ({a.room}) overlaps {b.name} ({b.room})")
    record("furniture_no_collision", not clashes, violations=clashes, exempt=exempt_pairs)
    violations.extend(clashes)

    # -- 3D containment: nothing pokes through a slab or a ceiling ---------- #
    containment = containment_report(spec, boxes)
    record("furniture_containment", not containment["violations"],
           violations=containment["violations"], items=containment["items"],
           ceiling_touch_tol_m=CEILING_TOUCH_TOL_M, ceiling_hug_max_m=CEILING_HUG_MAX_M,
           slab_sink_tol_m=SLAB_SINK_TOL_M)
    violations.extend(containment["violations"])

    # -- doors ------------------------------------------------------------- #
    zones = door_zones(spec, spec.wall_thickness)
    door_blocks: List[str] = []
    doors = {}
    for zone in zones:
        for room_id, rect in zone.rects.items():
            for box in boxes:
                if box.room != room_id or not box.blocks_walking:
                    continue
                if spec.floors and box.floor_id != zone.floor_id:
                    continue
                distance = rect_distance(box.aabb, rect)
                if distance < DOOR_CLEARANCE_M - 1e-9:
                    door_blocks.append(
                        f"{box.name} ({room_id}) is {distance:.3f} m from the door between "
                        f"{zone.a}/{zone.b} (min {DOOR_CLEARANCE_M} m)"
                    )
        label = f"{zone.a}|{zone.b}" if not spec.floors else f"{zone.floor_id}:{zone.a}|{zone.b}"
        doors[label] = {"center": list(zone.center), "width": zone.width,
                        "height": zone.height, "floor": zone.floor_id}
    record("doors_clear", not door_blocks, violations=door_blocks, doors=doors)
    violations.extend(door_blocks)

    # -- cameras ----------------------------------------------------------- #
    camera_reports = {}
    camera_bad: List[str] = []
    for cam in spec.cameras:
        room = rooms[cam.room]
        gap = wall_gap((cam.pos[0], cam.pos[1], cam.pos[0], cam.pos[1]), room.polygon)
        nearest, nearest_d = None, float("inf")
        for box in boxes:
            if box.room != cam.room or not box.blocks_camera:
                continue
            distance = rect_distance((cam.pos[0], cam.pos[1], cam.pos[0], cam.pos[1]), box.aabb)
            if distance < nearest_d:
                nearest, nearest_d = box.name, distance
        camera_reports[cam.name] = {"room": cam.room, "pos": list(cam.pos),
                                    "wall_gap_m": round(gap, 3),
                                    "nearest_furniture": nearest,
                                    "furniture_gap_m": round(nearest_d, 3)}
        if gap < CAMERA_WALL_GAP_M:
            camera_bad.append(f"camera '{cam.name}' is only {gap:.3f} m from a wall "
                              f"(min {CAMERA_WALL_GAP_M} m)")
        if not point_in_polygon(cam.pos, room.polygon):
            camera_bad.append(f"camera '{cam.name}' is outside room '{cam.room}'")
        if nearest_d < CAMERA_FURNITURE_GAP_M - 1e-9:
            camera_bad.append(f"camera '{cam.name}' is {nearest_d:.3f} m from '{nearest}' "
                              f"(min {CAMERA_FURNITURE_GAP_M} m)")
    record("cameras_clear", not camera_bad, violations=camera_bad, cameras=camera_reports)
    violations.extend(camera_bad)

    # -- cameras: a stand needs a real surface under it ---------------------- #
    stand_reports: Dict[str, dict] = {}
    stand_bad: List[str] = []
    for cam in spec.cameras:
        floor = spec.room_floor(cam.room)
        surfaces = spec.support_surfaces(cam.pos, floor.id if spec.floors else "")
        stair_surfaces = [surface for surface in surfaces
                          if surface.kind in ("step", "landing", "ramp")]
        below = [surface for surface in surfaces if surface.z <= cam.pos[2] + 1e-6]
        support = max(below, key=lambda surface: surface.z) if below else None
        entry = {
            "room": cam.room, "z": round(float(cam.pos[2]), 3),
            "surfaces": [{"z": round(surface.z, 3), "kind": surface.kind, "label": surface.label}
                         for surface in surfaces],
            "support": None if support is None else {
                "z": round(support.z, 3), "kind": support.kind, "label": support.label},
            "eye_height_m": None,
        }
        if stair_surfaces:
            # a stair / ramp is a walk track, not floor: a stand over its footprint
            # is above the steps (or inside the solid flight) of a hole in the slab
            top = max(stair_surfaces, key=lambda surface: surface.z)
            gap = float(cam.pos[2]) - top.z
            where = f"{gap:.3f} m below the camera" if gap >= 0.0 else f"{-gap:.3f} m above the camera"
            entry["eye_height_m"] = None if support is None else round(float(cam.pos[2]) - support.z, 3)
            stand_bad.append(
                f"camera '{cam.name}' stands over the {top.kind} of a stair/ramp (surface at "
                f"z={top.z:.3f} m, {top.label}) at ({cam.pos[0]:.2f}, {cam.pos[1]:.2f}), {where}: "
                f"a stair is not a camera stand -- put it on the floor slab of its own storey"
            )
        elif support is None:
            stand_bad.append(
                f"camera '{cam.name}' has no surface under it at ({cam.pos[0]:.2f}, {cam.pos[1]:.2f}): "
                f"it stands over a slab opening / void of floor "
                f"'{floor.id if spec.floors else 'f0'}'"
            )
        else:
            eye = float(cam.pos[2]) - support.z
            entry["eye_height_m"] = round(eye, 3)
            if abs(eye - spec_mod.CAMERA_EYE_HEIGHT_M) > spec_mod.CAMERA_EYE_HEIGHT_TOL_M + 1e-9:
                stand_bad.append(
                    f"camera '{cam.name}' eye height is {eye:.3f} m over the floor of room "
                    f"'{support.label}' at z={support.z:.3f} m (expected "
                    f"{spec_mod.CAMERA_EYE_HEIGHT_M} m +- {spec_mod.CAMERA_EYE_HEIGHT_TOL_M} m)"
                )
        stand_reports[cam.name] = entry
    record("cameras_stand", not stand_bad, violations=stand_bad, cameras=stand_reports,
           eye_height_m=spec_mod.CAMERA_EYE_HEIGHT_M,
           eye_height_tol_m=spec_mod.CAMERA_EYE_HEIGHT_TOL_M)
    violations.extend(stand_bad)

    # -- circulation ------------------------------------------------------- #
    circulation = circulation_report(spec, boxes, zones, grid_m=grid_m)
    reach_bad: List[str] = []
    for room_id, stats in circulation["rooms"].items():
        if stats["reachable_cells"] < min(ROOM_REACH_MIN_CELLS, stats["free_cells"]) or \
                (stats["free_cells"] and stats["reachable_cells"] < ROOM_REACH_MIN_FRACTION * stats["free_cells"]):
            reach_bad.append(
                f"room '{room_id}': only {stats['reachable_cells']} of {stats['free_cells']} free cells "
                f"reachable from the first door"
            )
    for name, stats in circulation["cameras"].items():
        if not stats["reachable"]:
            reach_bad.append(f"camera '{name}' is not reachable from the first door")
    record("circulation_reachable", not reach_bad and circulation["start_reachable"],
           violations=reach_bad + ([circulation["start_note"]] if not circulation["start_reachable"] else []),
           rooms=circulation["rooms"], cameras=circulation["cameras"],
           start=circulation["start"], floors=circulation.get("floors"),
           tracks=circulation.get("tracks", []),
           grid={"step_m": grid_m, "free_cells": circulation["free_cells"],
                 "reachable_cells": circulation["reachable_cells"],
                 "stair_links": circulation.get("stair_links", 0)})
    violations.extend(reach_bad)

    # -- stairs: numbers, slab opening, headroom, clear mouth ---------------- #
    stairs = stair_report(spec)
    record("stairs_geometry", stairs["geometry_ok"], stairs=stairs["stairs"],
           violations=stairs["geometry_violations"])
    record("stairs_slab_openings", stairs["openings_ok"], stairs=stairs["stairs"],
           violations=stairs["opening_violations"])
    record("stairs_headroom", stairs["headroom_ok"], stairs=stairs["stairs"],
           violations=stairs["headroom_violations"], min_headroom_m=spec_mod.STAIR_HEADROOM_MIN_M)
    clear_stairs = stair_clearance(spec, boxes)
    record("stairs_clear", not clear_stairs, violations=clear_stairs, stairs=stairs["stairs"])
    violations.extend(stairs["geometry_violations"] + stairs["opening_violations"]
                      + stairs["headroom_violations"] + clear_stairs)

    # -- stairs / ramps along the walk track: rise, pitch, body clearance ------ #
    track = walk_track_report(spec)
    record("stairs_walk_track", track["pass"], tracks=track["tracks"], violations=track["violations"])
    violations.extend(track["violations"])

    # -- informational: same-XY items on different floors are not a clash ---- #
    cross_floor = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a.floor_id == b.floor_id and a.floor_z == b.floor_z:
                continue
            if a.aabb[2] > b.aabb[0] and b.aabb[2] > a.aabb[0] \
                    and a.aabb[3] > b.aabb[1] and b.aabb[3] > a.aabb[1]:
                cross_floor.append({"a": a.name, "b": b.name, "a_floor": a.floor_id,
                                    "b_floor": b.floor_id,
                                    "a_z": [round(a.z[0], 3), round(a.z[1], 3)],
                                    "b_z": [round(b.z[0], 3), round(b.z[1], 3)]})
    record("cross_floor_xy_ok_info", True, role="informational", pairs=cross_floor,
           note="same-XY items on different storeys share no 3D volume; this list is a sanity read")

    report = {
        "kind": "blender_agent_layout_checks",
        "pass": all(entry["pass"] for entry in checks),
        "checks": checks,
        "violations": violations,
        "furniture": {box.name: {"asset": box.asset, "room": box.room, "placement": box.placement,
                                 "aabb": [round(v, 3) for v in box.aabb],
                                 "z": [round(v, 3) for v in box.z]} for box in boxes},
    }
    return report


def room_connectivity(polygon: Sequence[Sequence[float]], boxes: Sequence[ItemBox],
                      start_points: Sequence[Sequence[float]], wall_thickness: float = 0.12,
                      grid_m: float = 0.20, dilation: float = FURNITURE_DILATION_M) -> dict:
    """Share of a single room's floor that stays walkable from its doors.

    Coarse (0.20 m) on purpose: ``layout_gen`` calls this while it places
    furniture to reject a layout that would fence off part of the room.
    """
    poly = inset_polygon(polygon, wall_thickness / 2.0)
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    x0, y0 = min(xs) - grid_m, min(ys) - grid_m
    nx = int(math.ceil((max(xs) + grid_m - x0) / grid_m)) + 1
    ny = int(math.ceil((max(ys) + grid_m - y0) / grid_m)) + 1
    free = bytearray(nx * ny)
    blocked = bytearray(nx * ny)
    for ix in range(nx):
        cx = x0 + ix * grid_m
        for iy in range(ny):
            cy = y0 + iy * grid_m
            if point_in_polygon((cx, cy), poly):
                free[ix * ny + iy] = 1
    for box in boxes:
        if not box.blocks_walking:
            continue
        rect = (box.aabb[0] - dilation, box.aabb[1] - dilation, box.aabb[2] + dilation, box.aabb[3] + dilation)
        for ix in range(nx):
            cx = x0 + ix * grid_m
            if cx < rect[0] or cx > rect[2]:
                continue
            for iy in range(ny):
                cy = y0 + iy * grid_m
                if rect[1] <= cy <= rect[3]:
                    blocked[ix * ny + iy] = 1
    free_total = sum(1 for index in range(nx * ny) if free[index] and not blocked[index])
    reachable = bytearray(nx * ny)
    for point in start_points:
        ix = int(round((float(point[0]) - x0) / grid_m))
        iy = int(round((float(point[1]) - y0) / grid_m))
        best = None
        for radius in range(0, 8):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < nx and 0 <= jy < ny and free[jx * ny + jy] and not blocked[jx * ny + jy]:
                        best = (jx, jy)
                        break
                if best:
                    break
            if best:
                break
        if best is None:
            continue
        queue = [best]
        reachable[best[0] * ny + best[1]] = 1
        while queue:
            cx, cy = queue.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = cx + dx, cy + dy
                if not (0 <= jx < nx and 0 <= jy < ny):
                    continue
                index = jx * ny + jy
                if reachable[index] or not free[index] or blocked[index]:
                    continue
                reachable[index] = 1
                queue.append((jx, jy))
    reach_total = sum(reachable)
    return {"free_cells": int(free_total), "reachable_cells": int(reach_total),
            "fraction": (reach_total / free_total) if free_total else 1.0,
            "grid_m": grid_m}


def _grid_cell(x0: float, y0: float, grid_m: float, nx: int, ny: int, x: float, y: float):
    ix = int(round((x - x0) / grid_m))
    iy = int(round((y - y0) / grid_m))
    if 0 <= ix < nx and 0 <= iy < ny:
        return ix, iy
    return None


def _nearest_free_cell(storeys: Sequence[dict], floor_position: Optional[int],
                       point: Sequence[float], free: bytearray,
                       blocked: bytearray) -> Optional[int]:
    """Global index of the free cell nearest ``point`` on one storey's grid.

    A walk track's entry / exit node joins the storey here; ``None`` when the
    point is off the grid or every cell around it is blocked (no legal stand).
    """
    if floor_position is None or not (0 <= floor_position < len(storeys)):
        return None
    storey = storeys[floor_position]
    cell = _grid_cell(storey["x0"], storey["y0"], storey["step"], storey["nx"], storey["ny"],
                      float(point[0]), float(point[1]))
    if cell is None:
        return None
    for radius in range(0, 9):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                jx, jy = cell[0] + dx, cell[1] + dy
                if not (0 <= jx < storey["nx"] and 0 <= jy < storey["ny"]):
                    continue
                index = storey["offset"] + jx * storey["ny"] + jy
                if free[index] and not blocked[index]:
                    return index
    return None


def circulation_report(spec: "spec_mod.SceneSpec", boxes: Sequence[ItemBox], zones: Sequence[DoorZone],
                       grid_m: float = GRID_M) -> dict:
    """Per-storey walkable grids plus the stair / ramp walk tracks, BFS from the first door.

    Single-storey (legacy) specs get exactly one grid, so the numbers match the
    pre-multi-floor checker.  On a multi-floor plan every storey has its own
    grid, whose free cells are real floor only: slab openings *and stair
    footprints* are holes, because a stair is not the storey's floor.  Each
    stair / ramp contributes its walk track (``spec.stair_track`` /
    ``ramp_track``) as extra nodes that carry their own height, joined to a
    storey grid at the entry node (lower floor) and the exit node (upper floor)
    only -- so the storeys connect exactly where the walker steps on and off.
    """
    storeys_meta = spec.floors or [_synthetic_check_floor(spec)]
    floor_index = {floor.id: position for position, floor in enumerate(storeys_meta)}
    storeys: List[dict] = []
    global_free = bytearray()
    global_blocked = bytearray()
    global_room: List[Optional[str]] = []
    offset = 0

    for floor in storeys_meta:
        rooms = spec.floor_rooms(floor.id) if spec.floors else list(spec.rooms)
        xs = [float(p[0]) for room in rooms for p in room.polygon]
        ys = [float(p[1]) for room in rooms for p in room.polygon]
        if not xs:
            continue
        x0, y0 = min(xs) - grid_m, min(ys) - grid_m
        nx = int(math.ceil((max(xs) + grid_m - x0) / grid_m)) + 1
        ny = int(math.ceil((max(ys) + grid_m - y0) / grid_m)) + 1
        free = bytearray(nx * ny)
        room_of: List[Optional[str]] = [None] * (nx * ny)
        for room in rooms:
            poly = inset_polygon(room.polygon, spec.wall_thickness / 2.0)
            for ix in range(nx):
                cx = x0 + ix * grid_m
                if cx < min(p[0] for p in poly) - grid_m or cx > max(p[0] for p in poly) + grid_m:
                    continue
                for iy in range(ny):
                    cy = y0 + iy * grid_m
                    if cy < min(p[1] for p in poly) - grid_m or cy > max(p[1] for p in poly) + grid_m:
                        continue
                    if point_in_polygon((cx, cy), poly):
                        free[ix * ny + iy] = 1
                        room_of[ix * ny + iy] = room.id
        for zone in zones:
            if spec.floors and zone.floor_id != floor.id:
                continue
            if not spec.floors and zone.floor_id:
                continue
            rect = corridor_rect(zone, spec.wall_thickness)
            for ix in range(nx):
                cx = x0 + ix * grid_m
                if cx < rect[0] or cx > rect[2]:
                    continue
                for iy in range(ny):
                    cy = y0 + iy * grid_m
                    if rect[1] <= cy <= rect[3]:
                        free[ix * ny + iy] = 1
        # slab openings and stair footprints are holes in the storey's floor: the
        # stair is a track of its own nodes (built below), not walkable floor, so
        # a stand above the stairwell is no longer read as a cell of this storey
        if spec.floors:
            holes = [opening.outline for opening in spec.slab_openings_for(floor.id)]
            holes += [plan.footprint for plan in spec.stair_plans() + spec.ramp_plans()
                      if floor.id in (plan.connector.from_floor, plan.connector.to_floor)]
            for polygon in holes:
                for ix in range(nx):
                    cx = x0 + ix * grid_m
                    for iy in range(ny):
                        cy = y0 + iy * grid_m
                        if point_in_polygon((cx, cy), polygon):
                            free[ix * ny + iy] = 0
                            room_of[ix * ny + iy] = None
        blocked = bytearray(nx * ny)
        for box in boxes:
            if not box.blocks_walking:
                continue
            if spec.floors and box.floor_id != floor.id:
                continue
            rect = (box.aabb[0] - FURNITURE_DILATION_M, box.aabb[1] - FURNITURE_DILATION_M,
                    box.aabb[2] + FURNITURE_DILATION_M, box.aabb[3] + FURNITURE_DILATION_M)
            for ix in range(nx):
                cx = x0 + ix * grid_m
                if cx < rect[0] or cx > rect[2]:
                    continue
                for iy in range(ny):
                    cy = y0 + iy * grid_m
                    if cy < rect[1] or cy > rect[3]:
                        continue
                    blocked[ix * ny + iy] = 1

        storeys.append({"id": floor.id, "elevation": float(floor.elevation), "x0": x0, "y0": y0,
                        "step": grid_m, "nx": nx, "ny": ny, "free": free, "blocked": blocked,
                        "room_of": room_of, "offset": offset})
        global_free += free
        global_blocked += blocked
        global_room += room_of
        offset += nx * ny

    # -- walk tracks of the stairs and ramps: nodes that carry their own height,
    #    joined to a storey grid at the entry and the exit node only ---------- #
    links: Dict[int, List[int]] = {}
    track_report: List[dict] = []
    track_links = 0
    track_base = len(global_free)  # the floor grids end here; track nodes follow
    for plan in (spec.stair_plans() + spec.ramp_plans()) if spec.floors else []:
        nodes = plan.walk_nodes()
        indices = []
        for _node in nodes:
            indices.append(len(global_free))
            global_free.append(1)
            global_blocked.append(0)
            global_room.append(None)
        for a, b in zip(indices, indices[1:]):
            links.setdefault(a, []).append(b)
            links.setdefault(b, []).append(a)
        entry_cell = _nearest_free_cell(storeys, floor_index.get(plan.connector.from_floor),
                                        nodes[0].point, global_free, global_blocked)
        exit_cell = _nearest_free_cell(storeys, floor_index.get(plan.connector.to_floor),
                                       nodes[-1].point, global_free, global_blocked)
        for node_index, cell in ((indices[0], entry_cell), (indices[-1], exit_cell)):
            if cell is None:
                continue
            links.setdefault(node_index, []).append(cell)
            links.setdefault(cell, []).append(node_index)
        linked = entry_cell is not None and exit_cell is not None
        track_links += 1 if linked else 0
        track_report.append({
            "kind": plan.connector.kind,
            "from_room": plan.connector.from_room, "to_room": plan.connector.to_room,
            "nodes": len(nodes),
            "base": [round(nodes[0].point[0], 3), round(nodes[0].point[1], 3), round(nodes[0].z, 3)],
            "head": [round(nodes[-1].point[0], 3), round(nodes[-1].point[1], 3),
                     round(nodes[-1].z, 3)],
            "entry_linked": entry_cell is not None, "exit_linked": exit_cell is not None,
        })

    room_stats = {room.id: {"free_cells": 0, "reachable_cells": 0} for room in spec.rooms}
    for floor_position, storey in enumerate(storeys):
        free, blocked, room_of = storey["free"], storey["blocked"], storey["room_of"]
        for index in range(len(free)):
            if free[index] and not blocked[index] and room_of[index] is not None:
                room_stats[room_of[index]]["free_cells"] += 1

    start = None
    start_note = ""
    start_position = 0
    if zones:
        start_position = floor_index.get(zones[0].floor_id, 0)
        storey = storeys[start_position] if storeys else None
        if storey is not None:
            start = _grid_cell(storey["x0"], storey["y0"], storey["step"], storey["nx"], storey["ny"],
                               zones[0].center[0], zones[0].center[1])
            if start is not None:
                start = (storey["offset"] + start[0] * storey["ny"] + start[1],)
    else:
        first = spec.cameras[0] if spec.cameras else None
        if first is not None:
            start_position = 0
            storey = storeys[start_position] if storeys else None
            if storey is not None:
                cell = _grid_cell(storey["x0"], storey["y0"], storey["step"], storey["nx"],
                                  storey["ny"], first.pos[0], first.pos[1])
                if cell is not None:
                    start = (storey["offset"] + cell[0] * storey["ny"] + cell[1],)

    start_reachable = False
    reachable = bytearray(len(global_free))
    if start is not None:
        start_index = start[0]
        if global_free[start_index] and not global_blocked[start_index]:
            start_reachable = True
            reachable[start_index] = 1
        else:
            storey = storeys[start_position]
            start_note = (f"the start cell at {zones[0].center if zones else start_index} is blocked "
                          f"(free={bool(global_free[start_index])})")
        queue = [start_index]
        while queue:
            index = queue.pop()
            for neighbour in _neighbours(index, storeys, links):
                if reachable[neighbour] or not global_free[neighbour] or global_blocked[neighbour]:
                    continue
                reachable[neighbour] = 1
                queue.append(neighbour)
    if start_reachable:
        for index in range(len(global_free)):
            if reachable[index] and global_room[index] is not None:
                room_stats[global_room[index]]["reachable_cells"] += 1

    camera_stats = {}
    for cam in spec.cameras:
        position = floor_index.get(spec.room(cam.room).floor_id or "", 0) if spec.floors else 0
        storey = storeys[position] if storeys else None
        stats = {"cell": None, "free": False, "blocked": False, "reachable": False,
                 "floor": storey["id"] if storey else ""}
        if storey is not None:
            cell = _grid_cell(storey["x0"], storey["y0"], storey["step"], storey["nx"], storey["ny"],
                              cam.pos[0], cam.pos[1])
            if cell is not None:
                index = storey["offset"] + cell[0] * storey["ny"] + cell[1]
                stats.update({
                    "cell": [cell[0], cell[1]],
                    "free": bool(global_free[index]),
                    "blocked": bool(global_blocked[index]),
                    "reachable": bool(reachable[index]),
                })
        camera_stats[cam.name] = stats
    for stats in room_stats.values():
        stats["reachable_fraction"] = round(stats["reachable_cells"] / stats["free_cells"], 3) \
            if stats["free_cells"] else 0.0
    shape = [storeys[0]["nx"], storeys[0]["ny"]] if storeys else [0, 0]
    return {
        "start": list(zones[0].center) if zones else None,
        "start_reachable": start_reachable,
        "start_note": start_note,
        "rooms": room_stats,
        "cameras": camera_stats,
        "free_cells": int(sum(global_free[:track_base])),
        "reachable_cells": int(sum(reachable[:track_base])),
        "grid_step_m": grid_m,
        "grid_shape": shape,
        "floors": {
            storey["id"]: {
                "elevation": storey["elevation"], "grid_shape": [storey["nx"], storey["ny"]],
                "free_cells": int(sum(storey["free"])),
                "walkable_cells": int(sum(1 for index in range(len(storey["free"]))
                                          if storey["free"][index] and not storey["blocked"][index])),
            }
            for storey in storeys
        },
        "tracks": track_report,
        "stair_links": track_links,
    }


def _neighbours(index: int, storeys: Sequence[dict], links: Dict[int, List[int]]):
    """4-neighbour cells of one flattened grid node plus its stair / ramp track links."""
    for storey in storeys:
        size = storey["nx"] * storey["ny"]
        if index < storey["offset"] + size:
            local = index - storey["offset"]
            ix, iy = divmod(local, storey["ny"])
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < storey["nx"] and 0 <= jy < storey["ny"]:
                    yield storey["offset"] + jx * storey["ny"] + jy
            break
    for neighbour in links.get(index, ()):
        yield neighbour


def _synthetic_check_floor(spec: "spec_mod.SceneSpec") -> "spec_mod.FloorSpec":
    return spec_mod.FloorSpec(id="", elevation=0.0, height=spec.wall_height,
                              slab_thickness=spec_mod.DEFAULT_SLAB_M)


def _footprint_openings(spec: "spec_mod.SceneSpec", plan) -> List["spec_mod.SlabOpeningSpec"]:
    return spec.slab_openings_for(plan.connector.to_floor)


def _footprint_opening_coverage(spec: "spec_mod.SceneSpec", plan, step: float = 0.10) -> dict:
    """Share of a stair footprint covered by the slab openings of the floor above."""
    openings = _footprint_openings(spec, plan)
    xs = [float(p[0]) for p in plan.footprint]
    ys = [float(p[1]) for p in plan.footprint]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    nx = max(2, int(math.ceil((x1 - x0) / step)) + 1)
    ny = max(2, int(math.ceil((y1 - y0) / step)) + 1)
    inside = total = 0
    for ix in range(nx):
        x = x0 + (x1 - x0) * ix / (nx - 1)
        for iy in range(ny):
            y = y0 + (y1 - y0) * iy / (ny - 1)
            if not point_in_polygon((x, y), plan.footprint):
                continue
            total += 1
            if any(point_in_polygon((x, y), opening.outline) for opening in openings):
                inside += 1
    return {"fraction": (inside / total) if total else 0.0, "samples": total,
            "openings": len(openings)}


def _obstruction_bottoms(spec: "spec_mod.SceneSpec", point: Sequence[float],
                         z: float) -> List[Tuple[float, str]]:
    """Every storey surface above ``point`` at height ``z``: (bottom_z, label).

    A slab counts where the point lies under a room of that storey and outside
    every slab opening of it; a ceiling plane counts only where the room really
    owns a ceiling -- the part of it that an upper storey's slab does not replace
    (``spec.ceiling_above``), which is the top storey's roof and the uncovered
    rest of a partially covered room.
    """
    out: List[Tuple[float, str]] = []
    for floor in spec.floors:
        for room in spec.floor_rooms(floor.id):
            if not point_in_polygon(point, room.polygon):
                continue
            ceiling = floor.elevation + spec.room_height(room.id)
            slab_bottom = floor.elevation - floor.slab_thickness
            # the storey's slab is in the way as soon as its top surface is at or
            # above the point (a point *inside* the slab reports a negative clearance)
            if floor.elevation >= z - 1e-9:
                if not any(point_in_polygon(point, opening.outline)
                           for opening in spec.slab_openings_for(floor.id)):
                    out.append((slab_bottom, f"slab of floor '{floor.id}'"))
            if ceiling >= z - 1e-9 and spec.ceiling_above(room.id, point):
                out.append((ceiling, f"ceiling of room '{room.id}'"))
    return out


def stair_report(spec: "spec_mod.SceneSpec") -> dict:
    """Acceptance numbers for every stair: steps, width, landing, opening, headroom."""
    stairs: List[dict] = []
    geometry_violations: List[str] = []
    opening_violations: List[str] = []
    headroom_violations: List[str] = []

    for plan in spec.stair_plans():
        label = f"stair {plan.connector.from_room}->{plan.connector.to_room}"
        entry: dict = {
            "from_room": plan.connector.from_room,
            "to_room": plan.connector.to_room,
            "from_floor": plan.connector.from_floor,
            "to_floor": plan.connector.to_floor,
            "switchback": plan.switchback,
            "risers": plan.risers,
            "riser_m": round(plan.riser_m, 4),
            "tread_m": round(plan.tread_m, 4),
            "pitch_deg": round(plan.pitch_deg, 2),
            "width_m": round(plan.width, 3),
            "run_m": round(plan.run, 3),
            "landing_m": round(plan.landing_m, 3),
            "rise_m": round(plan.rise_total, 3),
            "footprint": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in plan.footprint],
            "violations": [],
        }
        if plan.riser_m > spec_mod.STAIR_RISER_MAX_M + 1e-9:
            geometry_violations.append(f"{label}: riser {plan.riser_m:.3f} m > "
                                       f"{spec_mod.STAIR_RISER_MAX_M} m")
        if plan.tread_m < spec_mod.STAIR_TREAD_MIN_M - 1e-9:
            geometry_violations.append(f"{label}: tread {plan.tread_m:.3f} m < "
                                       f"{spec_mod.STAIR_TREAD_MIN_M} m")
        if plan.width < spec_mod.STAIR_WIDTH_MIN_M - 1e-9:
            geometry_violations.append(f"{label}: width {plan.width:.3f} m < "
                                       f"{spec_mod.STAIR_WIDTH_MIN_M} m")
        if plan.switchback and plan.landing_m < spec_mod.STAIR_LANDING_MIN_M - 1e-9:
            geometry_violations.append(f"{label}: landing {plan.landing_m:.3f} m < "
                                       f"{spec_mod.STAIR_LANDING_MIN_M} m")

        coverage = _footprint_opening_coverage(spec, plan)
        entry["opening_coverage"] = round(coverage["fraction"], 3)
        if coverage["fraction"] < 0.98:
            opening_violations.append(
                f"{label}: only {coverage['fraction'] * 100:.1f}% of the stair footprint is under a "
                f"slab opening of floor '{plan.connector.to_floor}' ({coverage['openings']} openings)"
            )

        clearances = []
        for box in plan.boxes:
            centre = _centroid(box["polygon"])
            for bottom, what in _obstruction_bottoms(spec, centre, box["z1"]):
                clearance = bottom - box["z1"]
                clearances.append(clearance)
                if clearance < spec_mod.STAIR_HEADROOM_MIN_M - 1e-9:
                    headroom_violations.append(
                        f"{label}: {clearance:.2f} m headroom over the {box['role']} surface at "
                        f"{box['z1']:.2f} m ({what})"
                    )
        entry["min_headroom_m"] = round(min(clearances), 3) if clearances else None
        entry["violations"] = [item for item in
                               geometry_violations + opening_violations + headroom_violations
                               if label in item]
        stairs.append(entry)

    return {
        "stairs": stairs,
        "geometry_ok": not geometry_violations,
        "openings_ok": not opening_violations,
        "headroom_ok": not headroom_violations,
        "geometry_violations": geometry_violations,
        "opening_violations": opening_violations,
        "headroom_violations": headroom_violations,
    }


def walk_track_report(spec: "spec_mod.SceneSpec") -> dict:
    """Step rise, pitch, monotonicity and body clearance along every walk track.

    Reviewed along the *trajectory*, not on the footprint: what the walker's feet
    meet from the entry node to the exit node.  A stair has to climb (never dip),
    keep every riser and the pitch inside the bands, and leave
    ``STAIR_HEADROOM_MIN_M`` of body clearance above every node; a ramp is held to
    ``RAMP_SLOPE_MAX`` in pitch terms.  Clearance comes from
    :func:`_obstruction_bottoms`, so the partial-ceiling rule reaches a walker's
    head as well as the stair's own boxes.
    """
    tracks: List[dict] = []
    violations: List[str] = []
    ramp_pitch_max = math.degrees(math.atan(spec_mod.RAMP_SLOPE_MAX))
    for connector, nodes in spec.walk_tracks():
        label = f"{connector.kind} {connector.from_room}->{connector.to_room}"
        rises = [b.z - a.z for a, b in zip(nodes, nodes[1:])]
        pitches: List[float] = []
        for a, b in zip(nodes, nodes[1:]):
            run = math.dist(a.point, b.point)
            pitches.append(math.degrees(math.atan2(b.z - a.z, run)) if run > 1e-9 else 90.0)
        clearances: List[float] = []
        for node in nodes:
            clearances.extend(bottom - node.z
                              for bottom, _what in _obstruction_bottoms(spec, node.point, node.z)
                              if bottom >= node.z + 1e-6)
        monotonic = all(rise >= -1e-9 for rise in rises)
        max_rise = max(rises) if rises else 0.0
        max_pitch = max(pitches) if pitches else 0.0
        min_clearance = min(clearances) if clearances else None
        entry = {
            "connector": label, "kind": connector.kind,
            "from_floor": connector.from_floor, "to_floor": connector.to_floor,
            "nodes": len(nodes),
            "base_z": round(nodes[0].z, 4) if nodes else None,
            "head_z": round(nodes[-1].z, 4) if nodes else None,
            "max_step_rise_m": round(max_rise, 4),
            "max_pitch_deg": round(max_pitch, 2),
            "min_clearance_m": None if min_clearance is None else round(min_clearance, 3),
            "monotonic": monotonic,
            "kinds": [node.kind for node in nodes],
            "violations": [],
        }
        if connector.kind == "stair" and max_rise > spec_mod.STAIR_RISER_MAX_M + 1e-9:
            entry["violations"].append(f"{label}: a step of the walk track rises {max_rise:.3f} m "
                                       f"(> {spec_mod.STAIR_RISER_MAX_M} m)")
        pitch_limit = spec_mod.STAIR_PITCH_MAX_DEG if connector.kind == "stair" else ramp_pitch_max
        if max_pitch > pitch_limit + 1e-9:
            entry["violations"].append(f"{label}: the walk track pitches {max_pitch:.1f} deg "
                                       f"(> {pitch_limit:.1f} deg)")
        if not monotonic:
            entry["violations"].append(f"{label}: the walk track dips on the way up "
                                       f"{[round(node.z, 3) for node in nodes]}")
        if not entry["kinds"] or entry["kinds"][0] != "floor" or entry["kinds"][-1] != "floor":
            entry["violations"].append(f"{label}: the walk track does not start and end on the "
                                       f"storey floors")
        if min_clearance is not None and min_clearance < spec_mod.STAIR_HEADROOM_MIN_M - 1e-9:
            entry["violations"].append(f"{label}: only {min_clearance:.2f} m of body clearance above "
                                       f"the walk track (min {spec_mod.STAIR_HEADROOM_MIN_M} m)")
        tracks.append(entry)
        violations.extend(entry["violations"])
    return {"pass": not violations, "tracks": tracks, "violations": violations}


def _centroid(polygon: Sequence[Sequence[float]]) -> Tuple[float, float]:
    n = len(polygon)
    area = 0.0
    cx = cy = 0.0
    for i in range(n):
        x0, y0 = float(polygon[i][0]), float(polygon[i][1])
        x1, y1 = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area /= 2.0
    if abs(area) < 1e-12:
        points = [(float(p[0]), float(p[1])) for p in polygon]
        return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)
    return (cx / (6.0 * area), cy / (6.0 * area))


def stair_clearance(spec: "spec_mod.SceneSpec", boxes: Sequence[ItemBox]) -> List[str]:
    """No furniture on the stair, in front of its first riser or its top step."""
    violations: List[str] = []
    plans = list(spec.stair_plans()) + list(spec.ramp_plans())
    for plan in plans:
        label = f"{plan.connector.kind} {plan.connector.from_room}->{plan.connector.to_room}"
        xs = [float(p[0]) for p in plan.footprint]
        ys = [float(p[1]) for p in plan.footprint]
        footprint = (min(xs), min(ys), max(xs), max(ys))
        for box in boxes:
            if box.floor_id not in (plan.connector.from_floor, plan.connector.to_floor) \
                    and spec.floors:
                continue
            if box.z[0] > plan.to_elevation + 1e-9 or box.z[1] < plan.from_elevation - 1e-9:
                continue
            if rect_distance(box.aabb, footprint) <= 0.0:
                violations.append(f"{box.name} ({box.room}) stands on the {label} footprint")
        for end, rect in (("bottom", plan.base_rect), ("top", plan.head_rect)):
            for box in boxes:
                if not box.blocks_walking:
                    continue
                if box.z[0] > plan.to_elevation + 1e-9 or box.z[1] < plan.from_elevation - 1e-9:
                    continue
                distance = rect_distance(box.aabb, rect)
                if distance < DOOR_CLEARANCE_M - 1e-9:
                    violations.append(
                        f"{box.name} ({box.room}) is {distance:.3f} m from the {end} of the {label} "
                        f"(min {DOOR_CLEARANCE_M} m)"
                    )
    return violations


def check_file(spec_path: str, assets_index: Optional[Dict[str, dict]] = None,
               grid_m: float = GRID_M) -> dict:
    spec = spec_mod.load_spec(spec_path)
    report = check_spec(spec, assets_index, grid_m=grid_m)
    report["spec"] = os.path.abspath(spec_path)
    return report


def _print_report(report: dict, label: str) -> None:
    print(f"checks: {label}")
    for entry in report["checks"]:
        print(f"  [{'PASS' if entry['pass'] else 'FAIL'}] {entry['name']}")
        if not entry["pass"]:
            for violation in entry.get("violations", []):
                print(f"         - {violation}")
    rooms = next((entry for entry in report["checks"] if entry["name"] == "circulation_reachable"), None)
    if rooms:
        for room_id, stats in rooms["rooms"].items():
            print(f"  room {room_id:>10s}: {stats['reachable_cells']:5d}/{stats['free_cells']:5d} free cells "
                  f"reachable ({stats['reachable_fraction'] * 100:.1f}%)")
    print(f"checks: {'PASS' if report['pass'] else 'FAIL'} ({len(report['violations'])} violations)")


def _final_geometry(spec, boxes, notes, source, args, authored=None) -> dict:
    """The ``--final-*`` mode: box-based rules on the geometry as built."""
    index = load_assets_index(args.assets_index) \
        if args.assets_index and os.path.isfile(args.assets_index) else None
    report = check_final_geometry(spec, boxes, index, grid_m=args.grid, authored=authored,
                                  source=source, notes=notes)
    report["notes"] = notes
    if not args.quiet:
        _print_report(report, f"{source} (final geometry, {len(boxes)} boxes)")
        containment = next((entry for entry in report["checks"]
                            if entry["name"] == "furniture_containment"), None)
        for item in (containment or {}).get("items", []):
            face = item["lowest_face"]
            above = "nothing" if face is None else f"{face['bottom']:.3f} m ({face['label']})"
            print(f"  {item['name']:>22s} {item['room']:>10s} "
                  f"z {item['z'][0]:6.2f}-{item['z'][1]:6.2f}  lowest body above: "
                  f"{above}  clearance {item['clearance_m']}")
        moved = report["final_geometry"]["moved"]
        print(f"checks: {moved} of {len(boxes)} boxes differ from the spec geometry")
        for note in notes:
            print(f"checks: note: {note}")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        if not args.quiet:
            print(f"checks: report written to {args.report}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check furniture placement and circulation of a scene spec")
    parser.add_argument("spec", nargs="?", default=None, help="scene spec JSON")
    parser.add_argument("--assets-index", default=DEFAULT_ASSETS_INDEX)
    parser.add_argument("--report", default=None, help="write the JSON report here")
    parser.add_argument("--grid", type=float, default=GRID_M, help="circulation grid step in metres")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--final-bboxes", default=None,
                        help="JSON box list of the geometry as built ({'items': [{'name', 'room', "
                             "'bbox': [x0, y0, z0, x1, y1, z1]}]}); checks those boxes instead of "
                             "the spec")
    parser.add_argument("--final-manifest", default=None,
                        help="build_manifest.json of a built scene; its spec and settle's bbox_z "
                             "give the boxes to re-check (spec may be omitted)")
    args = parser.parse_args(argv)

    final_path = args.final_manifest or args.final_bboxes
    if final_path:
        try:
            spec = spec_mod.load_spec(args.spec) if args.spec else None
        except spec_mod.SpecError as error:
            print(f"checks: spec is invalid: {error}")
            return 2
        index = load_assets_index(args.assets_index) \
            if args.assets_index and os.path.isfile(args.assets_index) else None
        try:
            spec, boxes, notes, source = load_final_boxes(final_path, spec, index)
        except (spec_mod.SpecError, OSError, ValueError) as error:
            print(f"checks: final geometry input is invalid: {error}")
            return 2
        authored = item_boxes(spec, index)[0]
        report = _final_geometry(spec, boxes, notes, source, args, authored=authored)
        if args.spec:
            report["spec"] = os.path.abspath(args.spec)
        return 0 if report["pass"] else 1

    if not args.spec:
        parser.error("a scene spec is required (or --final-manifest with a spec inside it)")
    try:
        spec = spec_mod.load_spec(args.spec)
    except spec_mod.SpecError as error:
        print(f"checks: spec is invalid: {error}")
        if args.report:
            os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
            with open(args.report, "w", encoding="utf-8") as handle:
                json.dump({"kind": "blender_agent_layout_checks", "pass": False,
                           "checks": [], "violations": [str(error)], "spec": args.spec}, handle, indent=2)
        return 2

    index = load_assets_index(args.assets_index) if args.assets_index and os.path.isfile(args.assets_index) else None
    report = check_spec(spec, index, grid_m=args.grid)
    report["spec"] = os.path.abspath(args.spec)

    if not args.quiet:
        _print_report(report, args.spec)

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        if not args.quiet:
            print(f"checks: report written to {args.report}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
