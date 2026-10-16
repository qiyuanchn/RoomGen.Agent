"""Scene-spec schema, validation and the PanoWorld format conventions.

Pure stdlib on purpose: this module is imported both by the Blender side
(``blender_agent/run_build.py``) and by the host side
(``blender_agent/export_scene.py``, ``tools/validate_blender_agent_p0.py``).
It must never import ``bpy``, ``numpy`` or ``PIL``.

Format conventions implemented here (single source of truth for the repo):

ERP (equirectangular) pixel -> ray, matching
``panoworld_lrm/utils.py::compute_plucmap_pano``::

    lon = (x + 0.5) / W * 2*pi - pi          # x = column
    lat = pi/2 - (y + 0.5) / H * pi          # y = row, row 0 is the zenith
    ray_local = (cos(lat)*sin(lon), -sin(lat), cos(lat)*cos(lon))   # OpenCV frame
    ray_world = c2w[:3, :3] @ ray_local,  origin = c2w[:3, 3]

Yaw-0 camera-to-world rotation (verified against
``examples/full_pipeline_demo_datas/scene0000/viewpoints/*/extrinsics.txt``)::

    R0 = [[0, 0, 1],
          [-1, 0, 0],
          [0, -1, 0]]        # camX=(0,-1,0) camY=(0,0,-1) camZ=(1,0,0)

i.e. the camera looks along world +X, image right is world -Y, image up is
world +Z.  A camera yaw psi is applied as ``R = Rz(psi) @ R0``.

Depth is stored as ``place_depth.png`` (uint16) with
``meters = png_value / scale``, scale = 1000 (1 mm), 0 reserved for invalid.
``place_depth_scale.txt`` holds the scale as plain text.

3D-FRONT replay (P3, see ``blender_agent/front3d.py``) extends this schema
without changing any of the above: furniture may reference ``front3d:<jid>``
(the model folder under ``3D-FUTURE-model``, placed with the dataset's own
position / quaternion / per-axis scale via the optional ``scale_xyz``), and a
top-level ``front3d`` block tells the Blender side to replay the shell from the
dataset's inline mesh triangles instead of extruding room polygons.

Multi-floor building contract (schema_version 2, frozen)
--------------------------------------------------------
A spec without ``schema_version`` / ``floors`` is the legacy single-storey
contract above and keeps parsing and building unchanged.  A two-storey spec
adds::

    "schema_version": 2,
    "floors": [{"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15},
               {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15}],
    "rooms": [{"id": "hall", "floor_id": "f0", ...,
               "height": 2.5},          // optional per-room net height (<= floor height)
              ...],
    "connectors": [{"kind": "door", "from_room": ..., "to_room": ..., "from_floor": "f0",
                    "to_floor": "f0", "width": 0.9, "height": 2.05, "center": [x, y]},
                   {"kind": "stair", "from_room": "hall", "to_room": "landing",
                    "from_floor": "f0", "to_floor": "f1", "width": 0.9,
                    "run": 2.6, "rise": 2.85, "steps": 16, "landing_m": 1.0,
                    "center": [x, y], "direction_deg": 90}],
    "slab_openings": [{"outline": [[x, y], ...], "floors": ["f1"]}]

Semantics frozen with the generator:

* ``floors[].elevation`` is the *walking surface* of that storey, ``height``
  its net height (floor to ceiling) and ``slab_thickness`` the slab **below**
  it, so the slab body spans ``[elevation - slab_thickness, elevation]``.
  One storey's ceiling is the next storey's slab: ``elevation_above -
  slab_thickness == elevation + height`` is the stacked case (no duplicated
  ceiling), a floor or room that is not covered from above gets its own
  ceiling/roof slab of the same thickness rule.  An upper slab that would
  reach below the lower ceiling plane is a :class:`SpecError`.
* every room names its ``floor_id``; wall segments are merged **within one
  floor only** (identical 2D outlines on two storeys never merge), and door /
  window cutters are applied per floor (a cutter only cuts the wall boxes of
  its own z band).
* ``connector`` (also accepted: ``connectors``) carries every connection:
  ``door`` / ``open_passage`` (same floor, optional ``center`` on the shared
  wall and ``height``, default 2.05 m) and ``stair`` / ``ramp`` (different
  floors, optional ``center`` + ``direction_deg`` measured CCW from +X, plus
  ``run`` per flight, ``rise`` = the height difference, ``steps`` risers and
  ``landing_m`` = 0 straight run / > 0 switchback with a half landing).
  ``slab_opening`` (also ``slab_openings``) punches one outline through the
  floor slab of every listed floor (stairwell / void).
* ``windows[].sill`` / ``height`` stay **relative to the room floor**.
* furniture and cameras carry **world** z: the bottom of a piece standing on
  its room's floor is the floor's ``elevation``, an explicit ``z`` is a world
  height, and everything the builder writes (manifest, extrinsics, settled
  poses) stays in world coordinates.
* **walkable ground is real surfaces only.**  A storey's floor slab is floor
  where a room of that storey covers the point and no ``slab_opening`` of that
  storey is punched there; a stair or ramp is *not* floor -- it is a walk track
  of nodes that carry their own height (:func:`stair_track` / :func:`ramp_track`),
  joined to the storey grids at its entry and exit nodes only.  A camera stands
  on a floor slab of its own storey at ``elevation + CAMERA_EYE_HEIGHT_M``
  (see :meth:`SceneSpec.support_surfaces`), never in mid air over a stairwell.
* **ceilings are replaced per area, not per room.**  Where a storey above has a
  slab bottom on the room's ceiling plane and a room covering the area, the slab
  *is* the ceiling for exactly that area; the rest of the room keeps its own
  ceiling slab (:meth:`SceneSpec.ceiling_regions`).  A ``slab_opening`` is a
  declared void and stays open in both bodies.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

PANO_WIDTH = 2048
PANO_HEIGHT = 1024
DEPTH_SCALE = 1000.0
DEPTH_MAX_METERS = 65535.0 / DEPTH_SCALE

# Yaw-0 camera-to-world rotation. See module docstring.
R0: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)

_EPS = 1e-9
_POINT_EPS = 1e-6

#: schema versions: 1 = legacy single storey (implicit), 2 = floors/connectors
SCHEMA_VERSION_LEGACY = 1
SCHEMA_VERSION_MULTIFLOOR = 2
SCHEMA_VERSION = SCHEMA_VERSION_MULTIFLOOR

#: slab thickness of the implicit single floor of a legacy spec (see shell.py)
DEFAULT_SLAB_M = 0.05

CONNECTOR_KINDS = ("door", "open_passage", "stair", "ramp")
DOOR_LIKE_KINDS = ("door", "open_passage")
VERTICAL_KINDS = ("stair", "ramp")
DEFAULT_DOOR_HEIGHT_M = 2.05

#: stair / ramp acceptance bands, shared with checks.py and the generator
STAIR_WIDTH_MIN_M = 0.80
STAIR_RISER_MAX_M = 0.19
STAIR_TREAD_MIN_M = 0.24
STAIR_LANDING_MIN_M = 0.80
STAIR_HEADROOM_MIN_M = 2.0
#: the steepest legal riser/tread pair (0.19 / 0.24) is 38.4 deg, so a walk track
#: that climbs steeper than this has a broken step sequence
STAIR_PITCH_MAX_DEG = 40.0
RAMP_WIDTH_MIN_M = 0.90
#: a ramp that climbs a whole storey needs ~23 m of run at the 1:12 accessibility
#: limit, so the first version bounds the slope loosely (a steep interior ramp)
#: and leaves long accessible ramps to split levels
RAMP_SLOPE_MAX = 0.30

#: eye height of a camera above the surface it stands on: the stands of
#: ``layout_gen`` place the camera at ``floor elevation + CAMERA_EYE_HEIGHT_M``
#: and ``checks`` measures the same offset downwards against the support surface
CAMERA_EYE_HEIGHT_M = 1.5
#: a camera whose feet are further than this off its support is not standing on it
CAMERA_EYE_HEIGHT_TOL_M = 0.25


class SpecError(ValueError):
    """Raised when the scene spec is malformed or geometrically inconsistent."""


def yaw_rotation(yaw_rad: float) -> List[List[float]]:
    """Camera-to-world rotation for a camera that looks along +X at yaw 0."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    rz = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    return [[sum(rz[i][k] * R0[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def extrinsics_matrix(pos: Sequence[float], yaw_rad: float = 0.0) -> List[List[float]]:
    """4x4 camera-to-world matrix written to ``extrinsics.txt``."""
    rot = yaw_rotation(yaw_rad)
    return [
        [rot[0][0], rot[0][1], rot[0][2], float(pos[0])],
        [rot[1][0], rot[1][1], rot[1][2], float(pos[1])],
        [rot[2][0], rot[2][1], rot[2][2], float(pos[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    """Signed area (positive for counter-clockwise vertices in XY)."""
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i][0], polygon[i][1]
        x1, y1 = polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Even-odd rule; boundary hits are not guaranteed either way."""
    x, y = point[0], point[1]
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i][0], polygon[i][1]
        x1, y1 = polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]
        if (y0 > y) != (y1 > y):
            t = (y - y0) / (y1 - y0)
            if x < x0 + t * (x1 - x0):
                inside = not inside
    return inside


def _dedup(values: Sequence[float], eps: float = 1e-6) -> List[float]:
    out: List[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > eps:
            out.append(v)
    return out


@dataclass
class WallSegment:
    """A merged, kept wall span along one room-boundary line.

    ``p0``/``p1`` are the *centre-line* endpoints (wall thickness straddles the
    line), ``direction`` is the unit vector from p0 to p1, ``owners`` lists the
    rooms whose boundary covers this span and ``interior`` is True when two or
    more rooms share it (partition wall instead of exterior wall).  ``floor_id``
    names the storey the span was merged within ("" for a legacy single floor);
    identical 2D outlines on two storeys stay separate segments.
    """

    p0: Tuple[float, float]
    p1: Tuple[float, float]
    direction: Tuple[float, float]
    owners: Tuple[str, ...]
    floor_id: str = ""

    @property
    def interior(self) -> bool:
        return len(self.owners) > 1

    @property
    def length(self) -> float:
        return math.dist(self.p0, self.p1)

    def contains_point(self, point: Sequence[float], eps: float = 1e-6) -> bool:
        """True when ``point`` lies on the segment (within ``eps``)."""
        dx, dy = point[0] - self.p0[0], point[1] - self.p0[1]
        along = dx * self.direction[0] + dy * self.direction[1]
        perp = -dx * self.direction[1] + dy * self.direction[0]
        if abs(perp) > eps:
            return False
        return -eps <= along <= self.length + eps


def build_wall_segments(rooms: Sequence["RoomSpec"]) -> List[WallSegment]:
    """Collect every room boundary edge and merge duplicate / collinear spans.

    Two rooms that share a wall do not need identical vertex lists (the demo's
    living/bedroom wall only partially overlaps), so edges are grouped by their
    infinite line and every covered sub-span within a line is emitted once with
    its full owner set.  Rooms are grouped by ``floor_id`` first: only walls of
    the *same* storey merge, so two stacked rooms with the same 2D outline keep
    their own (vertically separate) wall spans.
    """
    groups: Dict[str, List["RoomSpec"]] = {}
    for room in rooms:
        groups.setdefault(getattr(room, "floor_id", None) or "", []).append(room)

    merged: List[WallSegment] = []
    for floor_id in sorted(groups):
        lines: Dict[Tuple[float, float, float], List[Tuple[float, float, str]]] = {}
        line_frames: Dict[Tuple[float, float, float], Tuple[Tuple[float, float], Tuple[float, float]]] = {}

        for room in groups[floor_id]:
            poly = room.polygon
            n = len(poly)
            for i in range(n):
                ax, ay = float(poly[i][0]), float(poly[i][1])
                bx, by = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
                dx, dy = bx - ax, by - ay
                length = math.hypot(dx, dy)
                if length < _POINT_EPS:
                    raise SpecError(f"room '{room.id}' has a degenerate edge at vertex {i}")
                key, (ux, uy), (nx, ny), c = canonical_line((ax, ay), (bx, by))
                line_frames.setdefault(key, ((ux, uy), (nx, ny)))
                ta = ux * ax + uy * ay
                tb = ux * bx + uy * by
                lines.setdefault(key, []).append((min(ta, tb), max(ta, tb), room.id))

        segments: List[WallSegment] = []
        for key, spans in sorted(lines.items()):
            (ux, uy), (nx, ny) = line_frames[key]
            c = key[2]
            cuts = _dedup([t for span in spans for t in (span[0], span[1])])
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b - a <= 1e-6:
                    continue
                mid = 0.5 * (a + b)
                owners = tuple(sorted({room_id for t0, t1, room_id in spans if t0 - 1e-6 <= mid <= t1 + 1e-6}))
                if not owners:
                    # a gap between two covered spans of the same line: neither side
                    # of it is a room boundary, so there is no wall there (a plan
                    # whose rooms touch the line only at its ends would otherwise
                    # grow a phantom wall across the middle of a room)
                    continue
                p0 = (ux * a + nx * c, uy * a + ny * c)
                p1 = (ux * b + nx * c, uy * b + ny * c)
                segments.append(WallSegment(p0=p0, p1=p1, direction=(ux, uy), owners=owners,
                                            floor_id=floor_id))

        for seg in segments:
            if merged:
                prev = merged[-1]
                if prev.floor_id == seg.floor_id and prev.owners == seg.owners \
                        and math.dist(prev.p1, seg.p0) < 1e-6:
                    merged[-1] = WallSegment(p0=prev.p0, p1=seg.p1, direction=prev.direction,
                                             owners=prev.owners, floor_id=prev.floor_id)
                    continue
            merged.append(seg)
    return merged


def wall_segment_for_point(
    segments: Sequence[WallSegment], point: Sequence[float], room_id: str,
    tol: float = 1e-6, floor_id: Optional[str] = None,
) -> Optional[WallSegment]:
    """First segment owned by ``room_id`` that contains ``point``.

    ``floor_id`` (when given) restricts the match to one storey -- required on a
    multi-floor plan, where a point is on a wall of every storey at its XY.
    """
    for seg in segments:
        if floor_id is not None and seg.floor_id != floor_id:
            continue
        if room_id in seg.owners and seg.contains_point(point, tol):
            return seg
    return None


def canonical_line(p0: Sequence[float], p1: Sequence[float]):
    """Canonical description of the infinite line through two XY points.

    Returns ``(key, u, n, c)``: ``u`` is the direction, flipped so that
    ``(ux, uy) >= (0, 0)`` lexicographically (which makes the key order-free),
    ``n`` the left normal and ``c`` the offset, so that points on the line are
    ``p(t) = u * t + n * c``.
    """
    dx, dy = float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])
    length = math.hypot(dx, dy)
    if length < _POINT_EPS:
        raise SpecError("canonical_line needs two distinct points")
    ux, uy = dx / length, dy / length
    if (ux, uy) < (0.0, 0.0):
        ux, uy = -ux, -uy
    nx, ny = -uy, ux
    c = nx * float(p0[0]) + ny * float(p0[1])
    return (round(nx, 6), round(ny, 6), round(c, 6)), (ux, uy), (nx, ny), c


def canonical_line_key(p0: Sequence[float], p1: Sequence[float]):
    """Just the line key, see :func:`canonical_line`."""
    return canonical_line(p0, p1)[0]


def line_parameter(point: Sequence[float], direction: Sequence[float]) -> float:
    """Scalar position of ``point`` along ``direction`` (unit vector expected)."""
    return float(point[0]) * float(direction[0]) + float(point[1]) * float(direction[1])


def inward_normal(room_id: str, point: Sequence[float], spec: "SceneSpec", tol: float = 1e-3):
    """Unit normal pointing *into* ``room_id`` from a point on its boundary.

    Room polygons are validated counter-clockwise, so the inward side of an edge
    ``(a -> b)`` is its left normal ``(-dy, dx)``.
    """
    room = spec.room(room_id)
    polygon = room.polygon
    count = len(polygon)
    for index in range(count):
        ax, ay = float(polygon[index][0]), float(polygon[index][1])
        bx, by = float(polygon[(index + 1) % count][0]), float(polygon[(index + 1) % count][1])
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        along = (point[0] - ax) * ux + (point[1] - ay) * uy
        perp = -(point[0] - ax) * uy + (point[1] - ay) * ux
        if abs(perp) <= tol and -1e-6 <= along <= length + 1e-6:
            return (-uy, ux)
    raise SpecError(f"point {point} is not on the boundary of room '{room_id}'")


def polygon_clip(subject: Sequence[Sequence[float]], clipper: Sequence[Sequence[float]]) -> List[List[float]]:
    """Sutherland-Hodgman clip of ``subject`` by ``clipper`` (exact for convex clippers).

    Used for room-overlap tests (same floor: rooms must not overlap; different
    floors: the storey stacking rule).  The clipping polygon is expected to be
    counter-clockwise and convex -- the axis-aligned rectangles the generator
    emits are both; a non-convex clipper gives an approximate (larger) area,
    which fails safe for the overlap checks that use this.
    """
    output = [[float(p[0]), float(p[1])] for p in subject]
    clip = [[float(p[0]), float(p[1])] for p in clipper]
    if polygon_area(clip) < 0.0:
        clip.reverse()
    count = len(clip)
    for i in range(count):
        if not output:
            return []
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % count]
        ex, ey = bx - ax, by - ay

        def inside(point) -> bool:
            # left of the counter-clockwise clipper edge (b-a) x (p-a) >= 0
            return (point[1] - ay) * ex - (point[0] - ax) * ey >= -_POINT_EPS

        clipped: List[List[float]] = []
        for j in range(len(output)):
            current = output[j]
            previous = output[j - 1]
            current_in, previous_in = inside(current), inside(previous)
            if current_in:
                if not previous_in:
                    clipped.append(_edge_intersection(previous, current, (ax, ay), (ex, ey)))
                clipped.append(current)
            elif previous_in:
                clipped.append(_edge_intersection(previous, current, (ax, ay), (ex, ey)))
        output = clipped
    return output


def _edge_intersection(p, q, origin, direction) -> List[float]:
    ex, ey = direction
    dx, dy = q[0] - p[0], q[1] - p[1]
    denominator = dx * ey - dy * ex
    if abs(denominator) < 1e-12:
        return [float(q[0]), float(q[1])]
    t = ((origin[0] - p[0]) * ey - (origin[1] - p[1]) * ex) / denominator
    return [float(p[0] + t * dx), float(p[1] + t * dy)]


def polygon_overlap_area(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Area of the intersection of two polygons (m^2, 0.0 when disjoint)."""
    clipped = polygon_clip(a, b)
    if len(clipped) < 3:
        return 0.0
    return abs(polygon_area(clipped))


def polygons_overlap(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], eps: float = 1e-6) -> bool:
    return polygon_overlap_area(a, b) > eps


def _axis_rect(polygon: Sequence[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    """``(x0, y0, x1, y1)`` when the polygon is an axis-aligned rectangle, else None."""
    if len(polygon) != 4:
        return None
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    for x, y in ((xs[i], ys[i]) for i in range(4)):
        if not (abs(x - x0) <= 1e-6 or abs(x - x1) <= 1e-6):
            return None
        if not (abs(y - y0) <= 1e-6 or abs(y - y1) <= 1e-6):
            return None
    return (x0, y0, x1, y1)


def _merge_rect_cells(kept: Sequence[Sequence[bool]], xs: Sequence[float],
                      ys: Sequence[float]) -> List[List[List[float]]]:
    """Greedy merge of a kept-cell lattice into rectangles (covers every kept cell)."""
    nx = len(xs) - 1
    ny = len(ys) - 1
    open_runs: Dict[Tuple[int, int], int] = {}
    out: List[List[List[float]]] = []
    for j in range(ny):
        runs: List[Tuple[int, int]] = []
        i = 0
        while i < nx:
            if not kept[j][i]:
                i += 1
                continue
            start = i
            while i < nx and kept[j][i]:
                i += 1
            runs.append((start, i - 1))
        still = {run: open_runs.get(run, j) for run in runs}
        for run, j0 in open_runs.items():
            if run not in still:
                out.append([[xs[run[0]], ys[j0]], [xs[run[1] + 1], ys[j0]],
                            [xs[run[1] + 1], ys[j]], [xs[run[0]], ys[j]]])
        open_runs = still
    for run, j0 in open_runs.items():
        out.append([[xs[run[0]], ys[j0]], [xs[run[1] + 1], ys[j0]],
                    [xs[run[1] + 1], ys[ny]], [xs[run[0]], ys[ny]]])
    return out


def polygon_difference(subject: Sequence[Sequence[float]],
                       cutters: Sequence[Sequence[Sequence[float]]]) -> List[List[List[float]]]:
    """``subject`` minus the union of ``cutters``, as a list of polygons (all inside it).

    Exact for axis-aligned rectangles -- what the generator and the hand-written
    demos emit -- by cutting the subject's lattice at every cutter edge and merging
    the surviving cells back into rectangles.  Any other outline keeps the whole
    subject, which is the safe side for a shell part: a piece that is not needed is
    a duplicated body, a missing one is a hole in the building.
    """
    subject_rect = _axis_rect(subject)
    cutter_rects = [_axis_rect(cutter) for cutter in cutters]
    if subject_rect is None or any(rect is None for rect in cutter_rects):
        return [[[float(p[0]), float(p[1])] for p in subject]]
    x0, y0, x1, y1 = subject_rect
    xs = _dedup([x0, x1] + [min(max(rect[0], x0), x1) for rect in cutter_rects if rect]
                + [min(max(rect[2], x0), x1) for rect in cutter_rects if rect])
    ys = _dedup([y0, y1] + [min(max(rect[1], y0), y1) for rect in cutter_rects if rect]
                + [min(max(rect[3], y0), y1) for rect in cutter_rects if rect])
    nx, ny = len(xs) - 1, len(ys) - 1
    kept = [[True] * nx for _ in range(ny)]
    for j in range(ny):
        cy = 0.5 * (ys[j] + ys[j + 1])
        for i in range(nx):
            cx = 0.5 * (xs[i] + xs[i + 1])
            if any(rect[0] - 1e-9 <= cx <= rect[2] + 1e-9 and rect[1] - 1e-9 <= cy <= rect[3] + 1e-9
                   for rect in cutter_rects if rect):
                kept[j][i] = False
    return _merge_rect_cells(kept, xs, ys)


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class FurnitureSpec:
    """One placed asset: ``ph:<id>`` (or ``midi3d:``/``gen:``) on the floor with a Z rotation.

    ``rot_z_deg`` is measured counter-clockwise from -Y (the front convention of
    ``assets.py``): 0 faces -Y, 90 faces +X, 180 faces +Y, 270 faces -X.
    ``z`` overrides the bottom height (default 0 = on the floor); ``mount``
    "ceiling" instead lifts the asset so its top touches the ceiling.
    ``builtin:rug`` / ``builtin:table_lamp`` are generated props (see details.py);
    the rug needs ``size`` [length, width].

    ``front3d:<jid>`` items (P3) name a folder under ``3D-FUTURE-model`` and are
    placed with the dataset's own pose: ``pos`` / ``z`` / ``rot_z_deg`` are the
    dataset transform expressed in this repo's Z-up frame, and ``scale_xyz``
    carries the dataset's per-axis scale (3D-FRONT scales furniture
    non-uniformly, so the uniform ``scale`` stays 1.0 for those items).
    """

    asset: str
    room: str
    pos: List[float]
    rot_z_deg: float = 0.0
    z: Optional[float] = None
    mount: str = "floor"
    scale: float = 1.0
    name: Optional[str] = None
    size: Optional[List[float]] = None
    scale_xyz: Optional[List[float]] = None


@dataclass
class RoomSpec:
    id: str
    type: str
    polygon: List[List[float]]
    floor_color: List[float] = field(default_factory=lambda: [0.6, 0.6, 0.6])
    floor_id: Optional[str] = None
    height: Optional[float] = None


@dataclass
class FloorSpec:
    """One storey: walking surface at ``elevation``, slab below it.

    The slab body spans ``[elevation - slab_thickness, elevation]``; the storey's
    own ceiling is the slab of the storey above (or a roof slab of the same
    thickness when nothing covers it).
    """

    id: str
    elevation: float = 0.0
    height: float = 2.7
    slab_thickness: float = DEFAULT_SLAB_M

    @property
    def top(self) -> float:
        return self.elevation + self.height

    @property
    def slab_bottom(self) -> float:
        return self.elevation - self.slab_thickness


@dataclass
class ConnectorSpec:
    """A connection between two rooms: door / open passage / stair / ramp.

    ``kind`` decides which geometry fields matter: door-like connectors need a
    ``center`` on the shared wall (defaults to the middle of the shared span) and
    a ``height``; stairs and ramps need ``center``, a climbing direction
    (``direction_deg`` CCW from +X, or a ``direction`` 2-vector), ``run``,
    ``rise``, ``steps`` and ``landing_m``.
    """

    kind: str
    from_room: str
    to_room: str
    from_floor: str
    to_floor: str
    width: float
    height: Optional[float] = None
    center: Optional[List[float]] = None
    direction_deg: Optional[float] = None
    direction: Optional[List[float]] = None
    run: Optional[float] = None
    rise: Optional[float] = None
    steps: Optional[int] = None
    landing_m: float = 0.0
    index: int = -1
    source: str = "connector"  # "connector" | "opening" (derived from spec.openings)

    @property
    def door_like(self) -> bool:
        return self.kind in DOOR_LIKE_KINDS

    @property
    def vertical(self) -> bool:
        return self.kind in VERTICAL_KINDS

    def label(self) -> str:
        return f"{self.kind} {self.from_room}->{self.to_room}"


@dataclass
class SlabOpeningSpec:
    """An outline punched through the floor slab of every listed floor."""

    outline: List[List[float]]
    floors: List[str]

    def contains(self, point: Sequence[float]) -> bool:
        return point_in_polygon(point, self.outline)


@dataclass
class StairPlan:
    """Resolved stair geometry (world XY, steps as solid boxes from the lower floor).

    ``boxes`` are ready to build: solid blocks from the lower floor surface up to
    the walking surface, each one a ``{"polygon": 4 XY points, "z0", "z1",
    "role"}``.  ``base_point`` is where a walker enters on the lower floor,
    ``head_point`` where they step off onto the upper floor.
    """

    connector: ConnectorSpec
    from_elevation: float
    to_elevation: float
    center: Tuple[float, float]
    direction: Tuple[float, float]
    width: float
    run: float
    landing_m: float
    risers: int
    rise_total: float
    riser_m: float
    tread_m: float
    pitch_deg: float
    switchback: bool
    footprint: List[List[float]]
    boxes: List[dict]
    base_point: Tuple[float, float]
    head_point: Tuple[float, float]
    base_rect: Tuple[float, float, float, float]
    head_rect: Tuple[float, float, float, float]

    def walk_nodes(self) -> List["WalkNode"]:
        """The ordered walk track of this stair, see :func:`stair_track`."""
        return stair_track(self)


@dataclass
class RampPlan:
    """Resolved ramp geometry: a solid wedge from the lower to the upper floor."""

    connector: ConnectorSpec
    from_elevation: float
    to_elevation: float
    center: Tuple[float, float]
    direction: Tuple[float, float]
    width: float
    run: float
    rise_total: float
    slope: float
    pitch_deg: float
    footprint: List[List[float]]
    wedge: dict
    base_point: Tuple[float, float]
    head_point: Tuple[float, float]
    base_rect: Tuple[float, float, float, float]
    head_rect: Tuple[float, float, float, float]

    def walk_nodes(self) -> List["WalkNode"]:
        """The ordered walk track of this ramp, see :func:`ramp_track`."""
        return ramp_track(self)


@dataclass
class WalkNode:
    """One point of a walk track: where a walker stands and on what surface.

    ``z`` is the height of the surface itself (a floor slab, a stair step top, a
    half landing, an inclined ramp), never an eye height.  ``kind`` names that
    surface -- ``floor`` / ``step`` / ``landing`` / ``ramp`` -- and ``polygon``
    carries the XY outline the node stands for (stair boxes), so a point can be
    mapped back to the node under it.  Stair and ramp tracks start and end on the
    storey floors: only those two nodes join the floor graph.
    """

    point: Tuple[float, float]
    z: float
    kind: str
    label: str = ""
    polygon: Optional[List[List[float]]] = None


@dataclass
class SupportSurface:
    """A walkable surface found at one XY point, as seen from one storey."""

    z: float
    kind: str
    label: str


@dataclass
class OpeningSpec:
    kind: str
    between: List[str]
    center: List[float]
    width: float
    height: float


@dataclass
class WindowSpec:
    room: str
    wall: List[List[float]]
    sill: float
    height: float


@dataclass
class CameraSpec:
    name: str
    room: str
    pos: List[float]


@dataclass
class StyleSpec:
    sun_elevation_deg: float = 35.0
    sun_azimuth_deg: float = 210.0
    sky_strength: float = 1.0
    sun_energy: float = 3.0
    view_transform: str = "Standard"
    exposure: float = 0.0


@dataclass
class SceneSpec:
    unit: str
    wall_height: float
    wall_thickness: float
    rooms: List[RoomSpec]
    openings: List[OpeningSpec]
    windows: List[WindowSpec]
    cameras: List[CameraSpec]
    style: StyleSpec
    furniture: List[FurnitureSpec] = field(default_factory=list)
    materials: Dict[str, object] = field(default_factory=dict)
    details: Dict[str, object] = field(default_factory=dict)
    lighting: Dict[str, object] = field(default_factory=dict)
    front3d: Dict[str, object] = field(default_factory=dict)
    source_path: Optional[str] = None
    raw: Optional[dict] = None
    schema_version: int = SCHEMA_VERSION_LEGACY
    floors: List[FloorSpec] = field(default_factory=list)
    connectors: List[ConnectorSpec] = field(default_factory=list)
    slab_openings: List[SlabOpeningSpec] = field(default_factory=list)

    # -- storeys ------------------------------------------------------------ #
    @property
    def multifloor(self) -> bool:
        return bool(self.floors)

    def floor(self, floor_id: str) -> FloorSpec:
        for floor in self.floors:
            if floor.id == floor_id:
                return floor
        raise SpecError(f"unknown floor id: {floor_id}")

    def room_floor(self, room_id: str) -> FloorSpec:
        """Storey a room belongs to; the implicit ground storey for a legacy spec."""
        room = self.room(room_id)
        if self.floors:
            if not room.floor_id:
                raise SpecError(f"room '{room_id}' has no floor_id in a multi-floor spec")
            return self.floor(room.floor_id)
        return FloorSpec(id="", elevation=0.0, height=self.wall_height, slab_thickness=DEFAULT_SLAB_M)

    def room_elevation(self, room_id: str) -> float:
        return self.room_floor(room_id).elevation

    def room_height(self, room_id: str) -> float:
        """Net height of a room: its own override, else the storey height."""
        room = self.room(room_id)
        if room.height is not None:
            return float(room.height)
        return self.room_floor(room_id).height

    def room_top(self, room_id: str) -> float:
        return self.room_elevation(room_id) + self.room_height(room_id)

    def floor_rooms(self, floor_id: str) -> List[RoomSpec]:
        return [room for room in self.rooms if room.floor_id == floor_id]

    def slab_openings_for(self, floor_id: str) -> List[SlabOpeningSpec]:
        return [opening for opening in self.slab_openings if floor_id in opening.floors]

    def in_slab_opening(self, floor_id: str, point: Sequence[float]) -> bool:
        return any(opening.contains(point) for opening in self.slab_openings_for(floor_id))

    # -- walkable surfaces --------------------------------------------------- #
    def ceiling_regions(self, room_id: str) -> List[List[List[float]]]:
        """Ceiling outline(s) this room still owns, see :func:`room_ceiling_parts`."""
        return room_ceiling_parts(self, room_id)

    def ceiling_replaced_by(self, room_id: str) -> List[dict]:
        """Upper slabs that replace parts of this room's ceiling, see :func:`room_ceiling_covers`."""
        return room_ceiling_covers(self, room_id)

    def ceiling_above(self, room_id: str, point: Sequence[float]) -> bool:
        """True when ``point`` is under a ceiling this room owns, see :func:`room_ceiling_contains`."""
        return room_ceiling_contains(self, room_id, point)

    def support_surfaces(self, point: Sequence[float], floor_id: str) -> List[SupportSurface]:
        """Walkable surfaces at ``point`` on one storey, see :func:`point_support_surfaces`."""
        return point_support_surfaces(self, point, floor_id)

    def walk_tracks(self) -> List[Tuple[ConnectorSpec, List[WalkNode]]]:
        """Every stair / ramp as an ordered walk track, with the connector that made it."""
        tracks = [(plan.connector, stair_track(plan)) for plan in self.stair_plans()]
        tracks += [(plan.connector, ramp_track(plan)) for plan in self.ramp_plans()]
        return tracks

    # -- connections -------------------------------------------------------- #
    def connections(self) -> List[ConnectorSpec]:
        """Every connection: parsed connectors plus door-like ones derived from ``openings``.

        Legacy ``openings`` keep working (and are what a v1 spec carries); on a
        multi-floor spec both lists may be used, and a door declared in either is
        checked, cut and walked through the same way.
        """
        out = list(self.connectors)
        for index, opening in enumerate(self.openings):
            floor_a = self.room_floor(opening.between[0]).id if self.floors else ""
            floor_b = self.room_floor(opening.between[1]).id if self.floors else ""
            out.append(
                ConnectorSpec(
                    kind=opening.kind, from_room=opening.between[0], to_room=opening.between[1],
                    from_floor=floor_a, to_floor=floor_b, width=float(opening.width),
                    height=float(opening.height), center=[float(opening.center[0]), float(opening.center[1])],
                    index=index, source="opening",
                )
            )
        return out

    def door_connections(self) -> List[ConnectorSpec]:
        return [item for item in self.connections() if item.door_like]

    def stair_connections(self) -> List[ConnectorSpec]:
        return [item for item in self.connections() if item.vertical]

    def stair_plans(self) -> List[StairPlan]:
        return [stair_plan(self, item) for item in self.connections() if item.kind == "stair"]

    def ramp_plans(self) -> List[RampPlan]:
        return [ramp_plan(self, item) for item in self.connections() if item.kind == "ramp"]

    def room(self, room_id: str) -> RoomSpec:
        for room in self.rooms:
            if room.id == room_id:
                return room
        raise SpecError(f"unknown room id: {room_id}")

    def wall_segments(self) -> List[WallSegment]:
        return build_wall_segments(self.rooms)

    def camera_adjacency(self) -> Dict[str, List[str]]:
        """Camera graph: two views are linked when their rooms share a door or a stair."""
        by_room: Dict[str, List[str]] = {room.id: [] for room in self.rooms}
        for cam in self.cameras:
            by_room[cam.room].append(cam.name)
        adjacency: Dict[str, List[str]] = {cam.name: [] for cam in self.cameras}
        for connector in self.connections():
            a, b = connector.from_room, connector.to_room
            for cam_a in by_room.get(a, []):
                for cam_b in by_room.get(b, []):
                    if cam_b not in adjacency[cam_a]:
                        adjacency[cam_a].append(cam_b)
                    if cam_a not in adjacency[cam_b]:
                        adjacency[cam_b].append(cam_a)
        return {name: sorted(peers) for name, peers in adjacency.items()}

    def room_adjacency(self) -> Dict[str, List[str]]:
        adjacency: Dict[str, List[str]] = {room.id: [] for room in self.rooms}
        for connector in self.connections():
            a, b = connector.from_room, connector.to_room
            if b not in adjacency[a]:
                adjacency[a].append(b)
            if a not in adjacency[b]:
                adjacency[b].append(a)
        return {room_id: sorted(peers) for room_id, peers in adjacency.items()}


# --------------------------------------------------------------------------- #
# stair / ramp geometry (pure, shared by the builder, checks and export)
# --------------------------------------------------------------------------- #
def _polygon_centroid(polygon: Sequence[Sequence[float]]) -> Tuple[float, float]:
    area = polygon_area(polygon)
    if abs(area) < 1e-12:
        points = [(float(p[0]), float(p[1])) for p in polygon]
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    cx = cy = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = float(polygon[i][0]), float(polygon[i][1])
        x1, y1 = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return (cx / (6.0 * area), cy / (6.0 * area))


def _polygon_bounds(polygon: Sequence[Sequence[float]]):
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _dominant_axis(polygon: Sequence[Sequence[float]]) -> Tuple[float, float]:
    x0, y0, x1, y1 = _polygon_bounds(polygon)
    return (1.0, 0.0) if (x1 - x0) >= (y1 - y0) else (0.0, 1.0)


def _unit_xy(vector, what: str) -> Tuple[float, float]:
    length = math.hypot(float(vector[0]), float(vector[1]))
    if length < 1e-9:
        raise SpecError(f"{what} direction is degenerate")
    return (float(vector[0]) / length, float(vector[1]) / length)


def _snap_axis(direction: Tuple[float, float]) -> Tuple[float, float]:
    """Snap a direction to the X/Y axis when it is within ~10 degrees of one."""
    dx, dy = direction
    if abs(dx) >= abs(dy) and abs(dx) >= 0.985:
        return (1.0 if dx > 0 else -1.0, 0.0)
    if abs(dy) > abs(dx) and abs(dy) >= 0.985:
        return (0.0, 1.0 if dy > 0 else -1.0)
    return direction


def _vertical_anchor(spec: "SceneSpec", connector: ConnectorSpec):
    """(center_xy, direction) of a stair/ramp: explicit, else derived from the plan.

    Derivation order: the slab opening that sits over the lower room (its centre
    and long axis), then the XY overlap of the two rooms (its centroid and long
    axis).  A spec may always state ``center`` / ``direction_deg`` explicitly,
    which is what a hand-written spec does.
    """
    lower = spec.room(connector.from_room)
    center = None
    direction = None
    if connector.center is not None:
        center = (float(connector.center[0]), float(connector.center[1]))
    if connector.direction is not None:
        direction = _snap_axis(_unit_xy(connector.direction, connector.label()))
    elif connector.direction_deg is not None:
        angle = math.radians(float(connector.direction_deg))
        direction = _snap_axis((math.cos(angle), math.sin(angle)))

    if center is None or direction is None:
        candidates = []
        for opening in spec.slab_openings_for(connector.to_floor):
            cx, cy = _polygon_centroid(opening.outline)
            if point_in_polygon((cx, cy), lower.polygon):
                candidates.append(opening.outline)
        if candidates:
            outline = max(candidates, key=lambda poly: abs(polygon_area(poly)))
            center = center or _polygon_centroid(outline)
            direction = direction or _dominant_axis(outline)
    if center is None or direction is None:
        overlap = polygon_clip(lower.polygon, spec.room(connector.to_room).polygon)
        if len(overlap) >= 3 and abs(polygon_area(overlap)) > 0.5:
            center = center or _polygon_centroid(overlap)
            direction = direction or _dominant_axis(overlap)
    if center is None:
        raise SpecError(f"{connector.label()}: no stair centre -- give 'center' or a slab opening "
                        f"over room '{connector.from_room}'")
    if direction is None:
        direction = (1.0, 0.0)
    return center, direction


def _local_box(center, direction, u0: float, u1: float, v0: float, v1: float) -> List[List[float]]:
    """Rectangle in the stair's local frame -> world XY polygon (CCW)."""
    ux, uy = direction
    vx, vy = -uy, ux
    cx, cy = center
    corners = []
    for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
        corners.append([cx + ux * u + vx * v, cy + uy * u + vy * v])
    if polygon_area(corners) < 0.0:
        corners.reverse()
    return corners


def _rect_of(center, direction, u0, u1, v0, v1):
    x0, y0, x1, y1 = _polygon_bounds(_local_box(center, direction, u0, u1, v0, v1))
    return (x0, y0, x1, y1)


def stair_plan(spec: "SceneSpec", connector: ConnectorSpec) -> StairPlan:
    """Resolve a ``kind: "stair"`` connector into world geometry.

    Straight run (``landing_m == 0``): one flight along ``direction`` from the
    entry edge, ``steps`` risers of ``rise / steps``, treads of ``run / steps``.
    Switchback (``landing_m > 0``): two half-flights of ``steps / 2`` risers side
    by side (each ``width`` wide) that climb along the direction to a half-height
    landing at the far end and climb back on the other side, so the walker leaves
    the stair next to where they entered -- one storey higher.  Steps are solid
    blocks from the lower floor surface up to their walking surface.
    """
    if not spec.floors:
        raise SpecError(f"{connector.label()}: stairs need a multi-floor spec ('floors')")
    if connector.run is None or connector.rise is None or connector.steps is None:
        raise SpecError(f"{connector.label()}: stair needs 'run', 'rise' and 'steps'")
    lower_floor = spec.floor(connector.from_floor)
    upper_floor = spec.floor(connector.to_floor)
    rise_total = float(upper_floor.elevation) - float(lower_floor.elevation)
    if rise_total <= 0.0:
        raise SpecError(f"{connector.label()}: to_floor '{connector.to_floor}' is not above "
                        f"from_floor '{connector.from_floor}'")
    if abs(float(connector.rise) - rise_total) > 0.01:
        raise SpecError(f"{connector.label()}: rise {connector.rise:.3f} m does not match the floor "
                        f"difference {rise_total:.3f} m")
    risers = int(connector.steps)
    if risers < 3:
        raise SpecError(f"{connector.label()}: needs at least 3 steps")
    width = float(connector.width)
    run = float(connector.run)
    landing = float(connector.landing_m or 0.0)
    switchback = landing > 0.0
    if switchback and risers % 2:
        raise SpecError(f"{connector.label()}: a switchback stair needs an even number of steps "
                        f"(two half flights), got {risers}")
    center, direction = _vertical_anchor(spec, connector)
    riser_m = rise_total / risers
    if switchback:
        per_flight = run / (risers / 2.0)
        length = run + landing
        half = lower_floor.elevation + rise_total / 2.0
        u0 = -length / 2.0
        boxes: List[dict] = []
        for i in range(risers // 2):
            top = lower_floor.elevation + (i + 1) * riser_m
            boxes.append({"polygon": _local_box(center, direction, u0 + i * per_flight,
                                                u0 + (i + 1) * per_flight, 0.0, width),
                          "z0": lower_floor.elevation, "z1": top, "role": "flight_a"})
        boxes.append({"polygon": _local_box(center, direction, u0 + run, u0 + length, -width, width),
                      "z0": lower_floor.elevation, "z1": half, "role": "landing"})
        for i in range(risers // 2):
            top = lower_floor.elevation + rise_total / 2.0 + (risers / 2 - i) * riser_m
            boxes.append({"polygon": _local_box(center, direction, u0 + i * per_flight,
                                                u0 + (i + 1) * per_flight, -width, 0.0),
                          "z0": lower_floor.elevation, "z1": top, "role": "flight_b"})
        base_point = (
            center[0] + direction[0] * (u0 - 0.30) + (-direction[1]) * (width / 2.0),
            center[1] + direction[1] * (u0 - 0.30) + direction[0] * (width / 2.0),
        )
        head_point = (
            center[0] + direction[0] * (u0 - 0.30) + (-direction[1]) * (-width / 2.0),
            center[1] + direction[1] * (u0 - 0.30) + direction[0] * (-width / 2.0),
        )
        footprint = _local_box(center, direction, u0, u0 + length, -width, width)
        base_rect = _rect_of(center, direction, u0 - 0.80, u0, 0.0, width)
        head_rect = _rect_of(center, direction, u0 - 0.80, u0, -width, 0.0)
    else:
        tread = run / risers
        u0 = -run / 2.0
        boxes = []
        for i in range(risers):
            top = lower_floor.elevation + (i + 1) * riser_m
            boxes.append({"polygon": _local_box(center, direction, u0 + i * tread,
                                                u0 + (i + 1) * tread, -width / 2.0, width / 2.0),
                          "z0": lower_floor.elevation, "z1": top, "role": "flight"})
        per_flight = tread
        base_point = (center[0] + direction[0] * (u0 - 0.30), center[1] + direction[1] * (u0 - 0.30))
        u1 = run / 2.0
        head_point = (center[0] + direction[0] * (u1 + 0.30), center[1] + direction[1] * (u1 + 0.30))
        footprint = _local_box(center, direction, u0, u1, -width / 2.0, width / 2.0)
        base_rect = _rect_of(center, direction, u0 - 0.80, u0, -width / 2.0, width / 2.0)
        head_rect = _rect_of(center, direction, u1, u1 + 0.80, -width / 2.0, width / 2.0)
    return StairPlan(
        connector=connector, from_elevation=lower_floor.elevation, to_elevation=upper_floor.elevation,
        center=center, direction=direction, width=width, run=run, landing_m=landing,
        risers=risers, rise_total=rise_total, riser_m=riser_m, tread_m=per_flight,
        pitch_deg=math.degrees(math.atan2(riser_m, per_flight)), switchback=switchback,
        footprint=footprint, boxes=boxes, base_point=base_point, head_point=head_point,
        base_rect=base_rect, head_rect=head_rect,
    )


def ramp_plan(spec: "SceneSpec", connector: ConnectorSpec) -> RampPlan:
    """Resolve a ``kind: "ramp"`` connector into a solid inclined wedge."""
    if not spec.floors:
        raise SpecError(f"{connector.label()}: ramps need a multi-floor spec ('floors')")
    if connector.run is None or connector.rise is None:
        raise SpecError(f"{connector.label()}: ramp needs 'run' and 'rise'")
    lower_floor = spec.floor(connector.from_floor)
    upper_floor = spec.floor(connector.to_floor)
    rise_total = float(upper_floor.elevation) - float(lower_floor.elevation)
    if rise_total <= 0.0:
        raise SpecError(f"{connector.label()}: to_floor '{connector.to_floor}' is not above "
                        f"from_floor '{connector.from_floor}'")
    if abs(float(connector.rise) - rise_total) > 0.01:
        raise SpecError(f"{connector.label()}: rise {connector.rise:.3f} m does not match the floor "
                        f"difference {rise_total:.3f} m")
    run = float(connector.run)
    width = float(connector.width)
    slope = rise_total / run
    center, direction = _vertical_anchor(spec, connector)
    ux, uy = direction
    u0 = -run / 2.0
    u1 = run / 2.0
    footprint = _local_box(center, direction, u0, u1, -width / 2.0, width / 2.0)
    # wedge: bottom quad on the lower floor, top quad sloping up towards +direction
    bottom = [(float(cx), float(cy), float(lower_floor.elevation)) for cx, cy in footprint]
    top = []
    for cx, cy in footprint:
        u = (float(cx) - center[0]) * ux + (float(cy) - center[1]) * uy
        t = (u - u0) / (u1 - u0)
        top.append((float(cx), float(cy), float(lower_floor.elevation) + t * rise_total))
    base_point = (center[0] + ux * (u0 - 0.30), center[1] + uy * (u0 - 0.30))
    head_point = (center[0] + ux * (u1 + 0.30), center[1] + uy * (u1 + 0.30))
    return RampPlan(
        connector=connector, from_elevation=lower_floor.elevation, to_elevation=upper_floor.elevation,
        center=center, direction=direction, width=width, run=run, rise_total=rise_total,
        slope=slope, pitch_deg=math.degrees(math.atan(slope)), footprint=footprint,
        wedge={"bottom": bottom, "top": top}, base_point=base_point, head_point=head_point,
        base_rect=_rect_of(center, direction, u0 - 0.80, u0, -width / 2.0, width / 2.0),
        head_rect=_rect_of(center, direction, u1, u1 + 0.80, -width / 2.0, width / 2.0),
    )


# --------------------------------------------------------------------------- #
# walk tracks, support surfaces and ceilings
# --------------------------------------------------------------------------- #
def stair_track(plan: StairPlan) -> List[WalkNode]:
    """The walker's path up a stair, bottom-up, with the surface under every node.

    Nodes are the lower floor at the base point, the top of every step in
    climbing order, the half landing of a switchback and the upper floor at the
    head point.  Step boxes are already ordered along the climb (their tops rise
    from the entry), so sorting a role's boxes by ``z1`` keeps the walk order for
    the straight run and for both switchback flights; ``walk_nodes`` on the plan
    is the entry point used by the checker and the builder.
    """
    nodes = [WalkNode(point=plan.base_point, z=plan.from_elevation, kind="floor",
                      label=f"floor at the base of '{plan.connector.from_room}'")]
    by_role: Dict[str, List[dict]] = {}
    for box in plan.boxes:
        by_role.setdefault(box["role"], []).append(box)
    ordered: List[dict] = []
    for role in ("flight", "flight_a", "landing", "flight_b"):
        ordered.extend(sorted(by_role.get(role, []), key=lambda box: float(box["z1"])))
    for index, box in enumerate(ordered):
        centroid = _polygon_centroid(box["polygon"])
        nodes.append(WalkNode(
            point=(float(centroid[0]), float(centroid[1])), z=float(box["z1"]),
            kind="landing" if box["role"] == "landing" else "step",
            label=f"{box['role']} {index}",
            polygon=[[float(p[0]), float(p[1])] for p in box["polygon"]],
        ))
    nodes.append(WalkNode(point=plan.head_point, z=plan.to_elevation, kind="floor",
                          label=f"floor at the head of '{plan.connector.to_room}'"))
    return nodes


def ramp_track(plan: RampPlan) -> List[WalkNode]:
    """The walker's path up a ramp: lower floor, middle of the incline, upper floor."""
    middle = (0.5 * (plan.base_point[0] + plan.head_point[0]),
              0.5 * (plan.base_point[1] + plan.head_point[1]))
    return [
        WalkNode(point=plan.base_point, z=plan.from_elevation, kind="floor",
                 label=f"floor at the base of '{plan.connector.from_room}'"),
        WalkNode(point=middle, z=0.5 * (plan.from_elevation + plan.to_elevation), kind="ramp",
                 label="ramp"),
        WalkNode(point=plan.head_point, z=plan.to_elevation, kind="floor",
                 label=f"floor at the head of '{plan.connector.to_room}'"),
    ]


def stair_surface_at(plan: StairPlan, point: Sequence[float]) -> Optional[SupportSurface]:
    """The step top / half landing a stair provides at ``point`` (None off the stair)."""
    for box in plan.boxes:
        if point_in_polygon(point, box["polygon"]):
            return SupportSurface(z=float(box["z1"]),
                                  kind="landing" if box["role"] == "landing" else "step",
                                  label=box["role"])
    return None


def ramp_surface_at(plan: RampPlan, point: Sequence[float]) -> Optional[SupportSurface]:
    """The inclined surface a ramp provides at ``point`` (None off the ramp)."""
    if not point_in_polygon(point, plan.footprint):
        return None
    ux, uy = plan.direction
    u = (float(point[0]) - plan.center[0]) * ux + (float(point[1]) - plan.center[1]) * uy
    t = min(1.0, max(0.0, (u + plan.run / 2.0) / plan.run))
    return SupportSurface(z=plan.from_elevation + t * plan.rise_total, kind="ramp",
                          label=f"ramp {plan.connector.from_room}->{plan.connector.to_room}")


def point_support_surfaces(spec: "SceneSpec", point: Sequence[float],
                           floor_id: str) -> List[SupportSurface]:
    """Every walkable surface at ``point``, as seen from one storey.

    The storey's own floor slab counts where a room of that storey covers the
    point **and the slab is not opened** -- a stairwell or a declared void has no
    floor at its XY, which is what the old footprint rule got wrong.  Stairs and
    ramps contribute the surface of the box the point falls on (a step top, a half
    landing, the incline), so a checker can tell a stand on the floor from one in
    mid air, and a stair case from a floor.
    """
    out: List[SupportSurface] = []
    if spec.floors:
        floor = spec.floor(floor_id)
        rooms = spec.floor_rooms(floor_id)
        opened = spec.in_slab_opening(floor_id, point)
    else:
        floor = FloorSpec(id="", elevation=0.0, height=spec.wall_height,
                          slab_thickness=DEFAULT_SLAB_M)
        rooms = list(spec.rooms)
        opened = False
    for room in rooms:
        if not opened and point_in_polygon(point, room.polygon):
            out.append(SupportSurface(z=floor.elevation, kind="floor", label=room.id))
    for plan in spec.stair_plans():
        surface = stair_surface_at(plan, point)
        if surface is not None:
            out.append(surface)
    for plan in spec.ramp_plans():
        surface = ramp_surface_at(plan, point)
        if surface is not None:
            out.append(surface)
    return out


def room_ceiling_covers(spec: "SceneSpec", room_id: str) -> List[dict]:
    """Upper-storey slabs that replace parts of this room's ceiling.

    A storey above replaces a ceiling where its slab bottom sits exactly on the
    room's ceiling plane *and* one of its rooms covers the area.  Each entry is
    ``{"floor": id, "room": id, "polygon": clipped outline, "area_m2": area}`` --
    a list because an upper floor may cover the room only partly, so the rest of
    the ceiling still has to be built.
    """
    room = spec.room(room_id)
    floor = spec.room_floor(room_id)
    ceiling = floor.elevation + spec.room_height(room_id)
    covers: List[dict] = []
    for upper in spec.floors:
        if upper.elevation <= floor.elevation + 1e-9:
            continue
        if abs((upper.elevation - upper.slab_thickness) - ceiling) > 1e-6:
            continue
        for other in spec.floor_rooms(upper.id):
            area = polygon_overlap_area(room.polygon, other.polygon)
            if area <= 1e-6:
                continue
            covers.append({"floor": upper.id, "room": other.id,
                           "polygon": polygon_clip(room.polygon, other.polygon),
                           "area_m2": area})
    return covers


def room_ceiling_parts(spec: "SceneSpec", room_id: str) -> List[List[List[float]]]:
    """The ceiling outline(s) this room still owns: the room minus what is covered.

    Empty when an upper storey covers the whole room (that slab *is* the ceiling,
    no duplicated body), the full polygon for a legacy single-storey room or one
    that nothing covers, and one rectangle per surviving region for a partial
    cover -- the "lower 4x4, upper 2x2" case that used to lose the whole ceiling.
    A declared ``slab_opening`` above is *not* filled in: the void is explicit.
    """
    room = spec.room(room_id)
    covers = room_ceiling_covers(spec, room_id)
    if not covers:
        return [[[float(p[0]), float(p[1])] for p in room.polygon]]
    return polygon_difference(room.polygon, [cover["polygon"] for cover in covers])


def room_ceiling_contains(spec: "SceneSpec", room_id: str, point: Sequence[float]) -> bool:
    """True when ``point`` lies under a ceiling the room owns (replacement is not)."""
    return any(point_in_polygon(point, part) for part in room_ceiling_parts(spec, room_id))


# --------------------------------------------------------------------------- #
# loading / validation
# --------------------------------------------------------------------------- #
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpecError(message)


def _vec3(value, what: str) -> List[float]:
    _require(isinstance(value, (list, tuple)) and len(value) == 3, f"{what} must be a 3-element list")
    return [float(v) for v in value]


def _vec2(value, what: str) -> List[float]:
    _require(isinstance(value, (list, tuple)) and len(value) >= 2, f"{what} must be a 2-element list")
    return [float(value[0]), float(value[1])]


def parse_spec(data: dict, source_path: Optional[str] = None) -> SceneSpec:
    """Build a :class:`SceneSpec` from a parsed JSON dict and validate it."""
    _require(isinstance(data, dict), "scene spec must be a JSON object")
    for key in ("wall_height", "wall_thickness", "rooms", "cameras"):
        _require(key in data, f"scene spec is missing required key '{key}'")

    unit = str(data.get("unit", "meter"))
    _require(unit == "meter", f"only unit='meter' is supported, got '{unit}'")
    wall_height = float(data["wall_height"])
    wall_thickness = float(data["wall_thickness"])
    _require(wall_height > 0.5, "wall_height must be > 0.5 m")
    _require(wall_thickness > 0.0, "wall_thickness must be > 0")

    raw_floors = data.get("floors")
    if raw_floors is None:
        _require("schema_version" not in data or int(data.get("schema_version", 1)) == SCHEMA_VERSION_LEGACY,
                 "a schema_version 2 spec needs a 'floors' list")
        schema_version = SCHEMA_VERSION_LEGACY
        floors: List[FloorSpec] = []
    else:
        _require("schema_version" in data,
                 "'floors' need an explicit \"schema_version\": 2 (a spec without one is the "
                 "legacy single-storey contract)")
        schema_version = int(data.get("schema_version"))
        _require(schema_version == SCHEMA_VERSION_MULTIFLOOR,
                 f"floors need schema_version {SCHEMA_VERSION_MULTIFLOOR}, got {schema_version}")
        _require(isinstance(raw_floors, list) and raw_floors, "'floors' must be a non-empty list")
        floors = []
        for i, raw_floor in enumerate(raw_floors):
            floor_id = str(raw_floor["id"])
            elevation = float(raw_floor.get("elevation", 0.0))
            floor_height = float(raw_floor.get("height", wall_height))
            slab = float(raw_floor.get("slab_thickness", DEFAULT_SLAB_M))
            _require(1.0 <= floor_height <= 10.0, f"floors[{i}] ('{floor_id}') height must be in [1, 10] m")
            _require(0.0 < slab <= 0.6, f"floors[{i}] ('{floor_id}') slab_thickness must be in (0, 0.6] m")
            _require(-20.0 <= elevation <= 200.0, f"floors[{i}] ('{floor_id}') elevation is out of range")
            floors.append(FloorSpec(id=floor_id, elevation=elevation, height=floor_height,
                                    slab_thickness=slab))
        floor_ids = [floor.id for floor in floors]
        _require(len(set(floor_ids)) == len(floor_ids), "floor ids must be unique")
        for a, b in zip(floors, floors[1:]):
            _require(b.elevation > a.elevation + 1e-6,
                     f"floors must be listed bottom-up: '{b.id}' is not above '{a.id}'")
    floor_id_set = {floor.id for floor in floors}

    rooms: List[RoomSpec] = []
    for i, raw_room in enumerate(data["rooms"]):
        room_id = str(raw_room["id"])
        polygon = [_vec2(p, f"rooms[{i}].polygon vertex") for p in raw_room["polygon"]]
        _require(len(polygon) >= 3, f"room '{room_id}' needs at least 3 vertices")
        color = raw_room.get("floor_color", [0.6, 0.6, 0.6])
        _require(len(color) == 3, f"room '{room_id}' floor_color must have 3 components")
        room_floor_id = raw_room.get("floor_id")
        if floors:
            _require(room_floor_id is not None, f"room '{room_id}' needs a floor_id "
                                                f"(specs with 'floors' require it)")
            _require(str(room_floor_id) in floor_id_set,
                     f"room '{room_id}' references unknown floor '{room_floor_id}'")
            room_floor_id = str(room_floor_id)
        else:
            _require(room_floor_id is None, f"room '{room_id}' has a floor_id but the spec has no 'floors'")
        room_height = raw_room.get("height")
        if room_height is not None:
            _require(floors, f"room '{room_id}' overrides 'height' -- only supported with 'floors'")
            room_height = float(room_height)
        rooms.append(
            RoomSpec(id=room_id, type=str(raw_room.get("type", "room")), polygon=polygon,
                     floor_color=[float(c) for c in color], floor_id=room_floor_id,
                     height=room_height)
        )
    ids = [room.id for room in rooms]
    _require(len(set(ids)) == len(ids), "room ids must be unique")
    for room in rooms:
        _require(polygon_area(room.polygon) > 0.0,
                 f"room '{room.id}' polygon must be counter-clockwise with positive area")
    room_by_id = {room.id: room for room in rooms}
    floor_by_id = {floor.id: floor for floor in floors}

    def _room_height(room_id: str) -> float:
        """Net room height: the room override, else the storey height."""
        room = room_by_id[room_id]
        if room.height is not None:
            return float(room.height)
        return wall_height if not floors else floor_by_id[room.floor_id].height

    openings: List[OpeningSpec] = []
    for i, raw in enumerate(data.get("openings", [])):
        between = [str(r) for r in raw["between"]]
        _require(len(between) == 2, f"openings[{i}].between must list exactly two rooms")
        for room_id in between:
            _require(room_id in ids, f"openings[{i}] references unknown room '{room_id}'")
        if floors:
            floors_of = {room_by_id[room_id].floor_id for room_id in between}
            _require(len(floors_of) == 1,
                     f"openings[{i}] joins rooms on different floors -- use a stair/ramp connector")
        headroom = max(_room_height(room_id) for room_id in between)
        width, height = float(raw["width"]), float(raw["height"])
        _require(width > 0.0 and height > 0.0, f"openings[{i}] needs positive width/height")
        _require(height <= headroom + _EPS,
                 f"openings[{i}].height exceeds the storey height ({headroom} m)")
        openings.append(OpeningSpec(kind=str(raw.get("kind", "door")), between=between,
                                    center=_vec2(raw["center"], f"openings[{i}].center"),
                                    width=width, height=height))

    windows: List[WindowSpec] = []
    for i, raw in enumerate(data.get("windows", [])):
        room_id = str(raw["room"])
        _require(room_id in ids, f"windows[{i}] references unknown room '{room_id}'")
        wall = [_vec2(p, f"windows[{i}].wall endpoint") for p in raw["wall"]]
        _require(len(wall) == 2, f"windows[{i}].wall must have exactly two endpoints")
        sill, height = float(raw["sill"]), float(raw["height"])
        _require(sill >= 0.0 and height > 0.0, f"windows[{i}] needs sill >= 0 and height > 0")
        _require(sill + height <= _room_height(room_id) + _EPS,
                 f"windows[{i}] top edge exceeds the room height ({_room_height(room_id)} m)")
        windows.append(WindowSpec(room=room_id, wall=wall, sill=sill, height=height))

    cameras: List[CameraSpec] = []
    for i, raw in enumerate(data["cameras"]):
        cameras.append(CameraSpec(name=str(raw["name"]), room=str(raw["room"]),
                                  pos=_vec3(raw["pos"], f"cameras[{i}].pos")))
    names = [cam.name for cam in cameras]
    _require(len(set(names)) == len(names), "camera names must be unique")
    _require(len(cameras) >= 1, "at least one camera is required")

    raw_style = data.get("style", {})
    style = StyleSpec(
        sun_elevation_deg=float(raw_style.get("sun_elevation_deg", 35.0)),
        sun_azimuth_deg=float(raw_style.get("sun_azimuth_deg", 210.0)),
        sky_strength=float(raw_style.get("sky_strength", 1.0)),
        sun_energy=float(raw_style.get("sun_energy", 3.0)),
        view_transform=str(raw_style.get("view_transform", "Standard")),
        exposure=float(raw_style.get("exposure", 0.0)),
    )
    _require(0.0 < style.sun_elevation_deg < 90.0, "style.sun_elevation_deg must be in (0, 90)")
    _require(style.sky_strength > 0.0, "style.sky_strength must be > 0")
    _require(style.sun_energy >= 0.0, "style.sun_energy must be >= 0")

    furniture: List[FurnitureSpec] = []
    for i, raw in enumerate(data.get("furniture", [])):
        label = f"furniture[{i}]"
        asset = str(raw["asset"])
        _require(":" in asset, f"{label}.asset must be 'ph:<polyhaven id>', 'midi3d:<id>', "
                               "'gen:<generated asset id>', 'builtin:<prop>' or 'front3d:<jid>'")
        scheme, _, asset_id = asset.partition(":")
        _require(scheme in ("ph", "midi3d", "gen", "builtin", "front3d"),
                 f"{label}.asset scheme '{scheme}' is unknown")
        _require(bool(asset_id), f"{label}.asset is missing the asset id")
        room_id = str(raw["room"])
        _require(room_id in ids, f"{label} references unknown room '{room_id}'")
        mount = str(raw.get("mount", "floor"))
        _require(mount in ("floor", "ceiling", "wall"), f"{label}.mount must be floor/ceiling/wall")
        scale = float(raw.get("scale", 1.0))
        _require(scale > 0.0, f"{label}.scale must be > 0")
        size = raw.get("size")
        scale_xyz = raw.get("scale_xyz")
        if scale_xyz is not None:
            _require(isinstance(scale_xyz, (list, tuple)) and len(scale_xyz) == 3,
                     f"{label}.scale_xyz must be a 3-element list")
            scale_xyz = [float(v) for v in scale_xyz]
            _require(all(v > 0.0 for v in scale_xyz), f"{label}.scale_xyz must be > 0 on every axis")
        furniture.append(
            FurnitureSpec(
                asset=asset,
                room=room_id,
                pos=_vec2(raw["pos"], f"{label}.pos"),
                rot_z_deg=float(raw.get("rot_z_deg", 0.0)),
                z=(float(raw["z"]) if raw.get("z") is not None else None),
                mount=mount,
                scale=scale,
                name=(str(raw["name"]) if raw.get("name") else None),
                size=([float(v) for v in size] if size else None),
                scale_xyz=scale_xyz,
            )
        )

    materials = data.get("materials", {}) or {}
    for key in ("floor", "wall", "rug", "bedding"):
        if materials.get(key):
            _require(str(materials[key]).startswith("ph:"), f"materials.{key} must be 'ph:<texture id>'")
    details = data.get("details", {}) or {}
    lighting = data.get("lighting", {}) or {}

    raw_connectors = data.get("connectors", data.get("connector"))
    connectors: List[ConnectorSpec] = []
    if raw_connectors is not None:
        _require(floors, "connectors need a multi-floor spec ('floors' and schema_version 2)")
        _require(isinstance(raw_connectors, list), "'connectors' must be a list")
        for i, raw in enumerate(raw_connectors):
            label = f"connectors[{i}]"
            kind = str(raw.get("kind", "door"))
            _require(kind in CONNECTOR_KINDS,
                     f"{label}.kind '{kind}' is unknown (door/open_passage/stair/ramp)")
            from_room, to_room = str(raw["from_room"]), str(raw["to_room"])
            for room_id in (from_room, to_room):
                _require(room_id in ids, f"{label} references unknown room '{room_id}'")
            from_floor = str(raw.get("from_floor", room_by_id[from_room].floor_id))
            to_floor = str(raw.get("to_floor", room_by_id[to_room].floor_id))
            _require(from_floor in floor_id_set, f"{label}.from_floor '{from_floor}' is unknown")
            _require(to_floor in floor_id_set, f"{label}.to_floor '{to_floor}' is unknown")
            _require(room_by_id[from_room].floor_id == from_floor,
                     f"{label}: room '{from_room}' is not on floor '{from_floor}'")
            _require(room_by_id[to_room].floor_id == to_floor,
                     f"{label}: room '{to_room}' is not on floor '{to_floor}'")
            if kind in DOOR_LIKE_KINDS:
                _require(from_floor == to_floor,
                         f"{label}: a {kind} joins rooms on different floors -- use stair/ramp")
            else:
                _require(floor_by_id[to_floor].elevation > floor_by_id[from_floor].elevation,
                         f"{label}: '{to_floor}' is not above '{from_floor}' (from = lower storey)")
            width = float(raw["width"]) if raw.get("width") is not None else (
                0.9 if kind in DOOR_LIKE_KINDS else 0.9)
            _require(width > 0.1, f"{label}.width must be > 0.1 m")
            height = float(raw["height"]) if raw.get("height") is not None else (
                min(DEFAULT_DOOR_HEIGHT_M, floor_by_id[from_floor].height - 0.10)
                if kind in DOOR_LIKE_KINDS else None)
            if height is not None:
                _require(height > 0.5 and height <= floor_by_id[from_floor].height + _EPS,
                         f"{label}.height must be in (0.5, storey height]")
            center = None
            if raw.get("center") is not None:
                center = _vec2(raw["center"], f"{label}.center")
            direction = None
            if raw.get("direction") is not None:
                direction = _vec2(raw["direction"], f"{label}.direction")
            direction_deg = raw.get("direction_deg")
            if direction_deg is not None:
                direction_deg = float(direction_deg)
                _require(math.isfinite(direction_deg), f"{label}.direction_deg must be finite")
            steps = raw.get("steps")
            if steps is not None:
                steps = int(steps)
                _require(steps >= 3, f"{label}.steps must be >= 3")
            run = float(raw["run"]) if raw.get("run") is not None else None
            rise = float(raw["rise"]) if raw.get("rise") is not None else None
            landing = float(raw.get("landing_m", 0.0))
            if kind == "stair":
                _require(run is not None and run > 0.3, f"{label}.run (per flight) must be > 0.3 m")
                _require(landing >= 0.0, f"{label}.landing_m must be >= 0")
            if kind == "ramp":
                _require(run is not None and run > 0.3, f"{label}.run must be > 0.3 m")
            connectors.append(ConnectorSpec(
                kind=kind, from_room=from_room, to_room=to_room, from_floor=from_floor,
                to_floor=to_floor, width=width, height=height, center=center,
                direction_deg=direction_deg, direction=direction, run=run, rise=rise,
                steps=steps, landing_m=landing, index=i,
            ))

    slab_openings: List[SlabOpeningSpec] = []
    raw_slab = data.get("slab_openings", data.get("slab_opening"))
    if raw_slab is not None:
        _require(floors, "slab openings need a multi-floor spec ('floors' and schema_version 2)")
        _require(isinstance(raw_slab, list), "'slab_openings' must be a list")
        for i, raw in enumerate(raw_slab):
            outline = [_vec2(p, f"slab_openings[{i}].outline vertex") for p in raw["outline"]]
            _require(len(outline) >= 3, f"slab_openings[{i}] outline needs at least 3 vertices")
            _require(polygon_area(outline) > 0.0,
                     f"slab_openings[{i}] outline must be counter-clockwise with positive area")
            floor_list = [str(f) for f in raw.get("floors", [])]
            _require(bool(floor_list), f"slab_openings[{i}] needs a 'floors' list")
            for floor_id in floor_list:
                _require(floor_id in floor_id_set,
                         f"slab_openings[{i}] references unknown floor '{floor_id}'")
            slab_openings.append(SlabOpeningSpec(outline=outline, floors=floor_list))

    raw_front3d = data.get("front3d") or {}
    _require(isinstance(raw_front3d, dict), "the 'front3d' block must be an object")
    if raw_front3d:
        _require(bool(raw_front3d.get("scene_json")), "front3d.scene_json is required")
        shell = str(raw_front3d.get("shell", "shell_mesh"))
        _require(shell == "shell_mesh",
                 f"front3d.shell '{shell}' is unknown (only 'shell_mesh', the dataset's own triangles)")

    spec = SceneSpec(unit=unit, wall_height=wall_height, wall_thickness=wall_thickness,
                     rooms=rooms, openings=openings, windows=windows, cameras=cameras,
                     style=style, furniture=furniture, materials=materials, details=details,
                     lighting=lighting, front3d=raw_front3d, source_path=source_path, raw=data,
                     schema_version=schema_version, floors=floors, connectors=connectors,
                     slab_openings=slab_openings)
    validate_spec(spec)
    return spec


def load_spec(path: str) -> SceneSpec:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return parse_spec(data, source_path=path)


def _shared_wall_span(spec: SceneSpec, segments, connector: ConnectorSpec):
    """Longest wall span owned by both rooms of a connector (its own floor)."""
    shared = [
        seg for seg in segments
        if seg.owners and connector.from_room in seg.owners and connector.to_room in seg.owners
        and (not spec.floors or seg.floor_id == connector.from_floor)
    ]
    if not shared:
        return None
    return max(shared, key=lambda seg: seg.length)


def validate_spec(spec: SceneSpec) -> None:
    """Geometric cross-checks that need the whole scene at once.

    Raises :class:`SpecError` on the first violation.
    """
    segments = spec.wall_segments()

    # -- storeys: one room minimum, no overlapping slabs, no double ceilings - #
    if spec.floors:
        for floor in spec.floors:
            _require(bool(spec.floor_rooms(floor.id)), f"floor '{floor.id}' has no rooms")
        for room in spec.rooms:
            height = spec.room_height(room.id)
            _require(0.5 <= height <= spec.room_floor(room.id).height + _EPS,
                     f"room '{room.id}' height {height} m must be in (0.5, storey height]")
        for floor in spec.floors:
            floor_rooms = spec.floor_rooms(floor.id)
            for i in range(len(floor_rooms)):
                for j in range(i + 1, len(floor_rooms)):
                    area = polygon_overlap_area(floor_rooms[i].polygon, floor_rooms[j].polygon)
                    _require(area <= 1e-6,
                             f"rooms '{floor_rooms[i].id}' and '{floor_rooms[j].id}' overlap on floor "
                             f"'{floor.id}' ({area:.3f} m^2)")
        # the slab rule: an upper storey slab may not reach below a lower room's
        # ceiling plane; when it is exactly the ceiling plane it *is* the ceiling
        for lower in spec.rooms:
            low_floor = spec.room_floor(lower.id)
            for upper in spec.rooms:
                up_floor = spec.room_floor(upper.id)
                if up_floor.elevation <= low_floor.elevation + 1e-9:
                    continue
                if not polygons_overlap(lower.polygon, upper.polygon):
                    continue
                ceiling = low_floor.elevation + spec.room_height(lower.id)
                slab_bottom = up_floor.elevation - up_floor.slab_thickness
                _require(slab_bottom >= ceiling - 1e-6,
                         f"floor '{up_floor.id}' slab (bottom {slab_bottom:.3f} m) overlaps room "
                         f"'{lower.id}' (ceiling {ceiling:.3f} m) -- upper slab and lower ceiling "
                         f"must not overlap")
                if slab_bottom > ceiling + 1e-6:
                    # a gap: the lower room needs its own ceiling slab, which must
                    # not run into the upper floor slab either
                    _require(ceiling + low_floor.slab_thickness <= slab_bottom + 1e-6,
                             f"room '{lower.id}' ceiling slab would overlap the slab of floor "
                             f"'{up_floor.id}'")

    # -- doors / open passages (every connector, legacy openings included) --- #
    for i, connector in enumerate(spec.door_connections()):
        label = f"connector[{i}] ({connector.kind} {connector.from_room}->{connector.to_room})"
        segment = _shared_wall_span(spec, segments, connector)
        _require(segment is not None,
                 f"{label}: no wall span shared by '{connector.from_room}' and '{connector.to_room}'"
                 + (f" on floor '{connector.from_floor}'" if spec.floors else ""))
        center = connector.center
        if center is None:
            center = [(segment.p0[0] + segment.p1[0]) / 2.0, (segment.p0[1] + segment.p1[1]) / 2.0]
            connector.center = center
        _require(segment.contains_point(center, 1e-6),
                 f"{label}: center {center} does not lie on the shared wall span")
        half = connector.width / 2.0
        along = (center[0] - segment.p0[0]) * segment.direction[0] \
            + (center[1] - segment.p0[1]) * segment.direction[1]
        _require(-1e-6 <= along - half and along + half <= segment.length + 1e-6,
                 f"{label}: opening width {connector.width} does not fit into the shared wall span "
                 f"(span length {segment.length:.4f} m, centre offset {along:.4f} m)")
        headroom = min(spec.room_height(connector.from_room), spec.room_height(connector.to_room))
        _require(connector.height <= headroom + _EPS,
                 f"{label}: height {connector.height} exceeds the room height ({headroom} m)")

    # -- windows: on a wall of the room's own storey, inside its height ------ #
    for i, window in enumerate(spec.windows):
        label = f"windows[{i}] (room {window.room})"
        room = spec.room(window.room)
        floor_id = room.floor_id if spec.floors else ""
        area = abs(polygon_area(room.polygon))
        for endpoint in window.wall:
            on_boundary = False
            n = len(room.polygon)
            for j in range(n):
                a = room.polygon[j]
                b = room.polygon[(j + 1) % n]
                dx, dy = b[0] - a[0], b[1] - a[1]
                length = math.hypot(dx, dy)
                ux, uy = dx / length, dy / length
                along = (endpoint[0] - a[0]) * ux + (endpoint[1] - a[1]) * uy
                perp = -(endpoint[0] - a[0]) * uy + (endpoint[1] - a[1]) * ux
                if abs(perp) <= 1e-6 and -1e-6 <= along <= length + 1e-6:
                    on_boundary = True
                    break
            _require(on_boundary, f"{label}: wall endpoint {endpoint} is not on the room boundary")
        middle = [(window.wall[0][0] + window.wall[1][0]) / 2.0, (window.wall[0][1] + window.wall[1][1]) / 2.0]
        seg = wall_segment_for_point(segments, middle, window.room, floor_id=floor_id)
        _require(seg is not None, f"{label}: window span is not on a wall of room '{window.room}'")
        length = math.dist(window.wall[0], window.wall[1])
        _require(length > 1e-3, f"{label}: window span is degenerate")
        _require(all(seg.contains_point(p, 1e-6) for p in window.wall),
                 f"{label}: window span crosses a wall junction or is longer than the wall span")
        _require(area > 0.0, f"{label}: room polygon has no area")
        _require(window.sill + window.height <= spec.room_height(window.room) + _EPS,
                 f"{label}: sill+height exceeds the room height")

    # -- cameras and furniture: world z inside the room's own storey band ---- #
    for i, cam in enumerate(spec.cameras):
        label = f"cameras[{i}] ('{cam.name}')"
        room = spec.room(cam.room)
        elevation = spec.room_elevation(cam.room)
        top = elevation + spec.room_height(cam.room)
        _require(point_in_polygon(cam.pos, room.polygon),
                 f"{label}: position {cam.pos} is outside room '{cam.room}'")
        _require(elevation < cam.pos[2] < top,
                 f"{label}: world z={cam.pos[2]} must be in ({elevation}, {top}) -- the storey band "
                 f"of room '{cam.room}'")

    for i, item in enumerate(spec.furniture):
        label = f"furniture[{i}] ('{item.asset}' in {item.room})"
        room = spec.room(item.room)
        elevation = spec.room_elevation(item.room)
        top = elevation + spec.room_height(item.room)
        _require(point_in_polygon(item.pos, room.polygon),
                 f"{label}: {item.pos} is outside room '{item.room}' (place furniture inside; "
                 f"the Blender side checks the bounding box against the walls)")
        if item.z is None and item.mount == "wall":
            raise SpecError(f"{label}: mount 'wall' needs an explicit z")
        if item.z is not None:
            _require(elevation - 0.01 <= item.z <= top,
                     f"{label}: world z={item.z} is outside room '{item.room}' "
                     f"(floor {elevation}, ceiling {top})")
        if item.asset == "builtin:rug":
            _require(item.size is not None and len(item.size) == 2 and min(item.size) > 0.2,
                     f"{label}: builtin:rug needs size [length, width] in metres")

    # -- stairs and ramps: resolve the geometry and check the shell contract - #
    for plan in spec.stair_plans():
        connector = plan.connector
        label = f"connector[{connector.index}] (stair {connector.from_room}->{connector.to_room})"
        _require(all(point_in_polygon(corner, spec.room(connector.from_room).polygon)
                     for corner in plan.footprint),
                 f"{label}: the stair footprint leaves room '{connector.from_room}' -- give a "
                 f"'center' that keeps it inside the lower room")
        _require(all(point_in_polygon(corner, spec.room(connector.to_room).polygon)
                     for corner in plan.footprint),
                 f"{label}: the stair footprint leaves room '{connector.to_room}' on the upper floor")
        _require(plan.riser_m <= STAIR_RISER_MAX_M + 1e-9,
                 f"{label}: riser {plan.riser_m:.3f} m exceeds {STAIR_RISER_MAX_M} m")
        _require(plan.tread_m >= STAIR_TREAD_MIN_M - 1e-9,
                 f"{label}: tread {plan.tread_m:.3f} m below {STAIR_TREAD_MIN_M} m -- increase run "
                 f"or lower the step count")
        _require(plan.width >= STAIR_WIDTH_MIN_M - 1e-9,
                 f"{label}: flight width {plan.width:.3f} m below {STAIR_WIDTH_MIN_M} m")
        if plan.switchback:
            _require(plan.landing_m >= STAIR_LANDING_MIN_M - 1e-9,
                     f"{label}: landing {plan.landing_m:.3f} m below {STAIR_LANDING_MIN_M} m")
        for opening in spec.slab_openings_for(connector.to_floor):
            if point_in_polygon(plan.center, opening.outline):
                break
        else:
            _require(False,
                     f"{label}: no slab opening on floor '{connector.to_floor}' covers the stair "
                     f"centre {plan.center} -- the stair would run into the slab")
    for plan in spec.ramp_plans():
        connector = plan.connector
        label = f"connector[{connector.index}] (ramp {connector.from_room}->{connector.to_room})"
        _require(plan.width >= RAMP_WIDTH_MIN_M - 1e-9,
                 f"{label}: ramp width {plan.width:.3f} m below {RAMP_WIDTH_MIN_M} m")
        _require(plan.slope <= RAMP_SLOPE_MAX + 1e-9,
                 f"{label}: slope {plan.slope * 100:.1f}% exceeds {RAMP_SLOPE_MAX * 100:.1f}%")


def dump_manifest_spec(spec: SceneSpec) -> dict:
    """The original spec dict, with a stable key order for the manifest."""
    if spec.raw is not None:
        return spec.raw
    return {
        "unit": spec.unit,
        "schema_version": spec.schema_version,
        "wall_height": spec.wall_height,
        "wall_thickness": spec.wall_thickness,
        "floors": [
            {"id": f.id, "elevation": f.elevation, "height": f.height, "slab_thickness": f.slab_thickness}
            for f in spec.floors
        ],
        "rooms": [
            {"id": r.id, "type": r.type, "polygon": r.polygon, "floor_color": r.floor_color,
             "floor_id": r.floor_id, "height": r.height}
            for r in spec.rooms
        ],
        "connectors": [
            {"kind": c.kind, "from_room": c.from_room, "to_room": c.to_room,
             "from_floor": c.from_floor, "to_floor": c.to_floor, "width": c.width,
             "height": c.height, "center": c.center, "direction_deg": c.direction_deg,
             "run": c.run, "rise": c.rise, "steps": c.steps, "landing_m": c.landing_m}
            for c in spec.connectors
        ],
        "slab_openings": [
            {"outline": o.outline, "floors": o.floors} for o in spec.slab_openings
        ],
        "openings": [
            {"kind": o.kind, "between": o.between, "center": o.center, "width": o.width, "height": o.height}
            for o in spec.openings
        ],
        "windows": [
            {"room": w.room, "wall": w.wall, "sill": w.sill, "height": w.height} for w in spec.windows
        ],
        "cameras": [{"name": c.name, "room": c.room, "pos": c.pos} for c in spec.cameras],
        "furniture": [
            {"asset": f.asset, "room": f.room, "pos": f.pos, "rot_z_deg": f.rot_z_deg,
             "z": f.z, "mount": f.mount, "scale": f.scale, "name": f.name,
             "scale_xyz": f.scale_xyz}
            for f in spec.furniture
        ],
        "materials": spec.materials,
        "details": spec.details,
        "lighting": spec.lighting,
        "front3d": spec.front3d,
        "style": {
            "sun_elevation_deg": spec.style.sun_elevation_deg,
            "sun_azimuth_deg": spec.style.sun_azimuth_deg,
            "sky_strength": spec.style.sky_strength,
            "sun_energy": spec.style.sun_energy,
            "view_transform": spec.style.view_transform,
            "exposure": spec.style.exposure,
        },
    }
