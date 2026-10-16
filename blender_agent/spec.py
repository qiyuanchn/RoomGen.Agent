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
    more rooms share it (partition wall instead of exterior wall).
    """

    p0: Tuple[float, float]
    p1: Tuple[float, float]
    direction: Tuple[float, float]
    owners: Tuple[str, ...]

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
    its full owner set.
    """
    lines: Dict[Tuple[float, float, float], List[Tuple[float, float, str]]] = {}
    line_frames: Dict[Tuple[float, float, float], Tuple[Tuple[float, float], Tuple[float, float]]] = {}

    for room in rooms:
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
            segments.append(WallSegment(p0=p0, p1=p1, direction=(ux, uy), owners=owners))

    merged: List[WallSegment] = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            if prev.owners == seg.owners and math.dist(prev.p1, seg.p0) < 1e-6:
                merged[-1] = WallSegment(p0=prev.p0, p1=seg.p1, direction=prev.direction, owners=prev.owners)
                continue
        merged.append(seg)
    return merged


def wall_segment_for_point(
    segments: Sequence[WallSegment], point: Sequence[float], room_id: str, tol: float = 1e-6
) -> Optional[WallSegment]:
    for seg in segments:
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

    def room(self, room_id: str) -> RoomSpec:
        for room in self.rooms:
            if room.id == room_id:
                return room
        raise SpecError(f"unknown room id: {room_id}")

    def wall_segments(self) -> List[WallSegment]:
        return build_wall_segments(self.rooms)

    def camera_adjacency(self) -> Dict[str, List[str]]:
        """Camera graph: two views are linked when their rooms share a door."""
        by_room: Dict[str, List[str]] = {room.id: [] for room in self.rooms}
        for cam in self.cameras:
            by_room[cam.room].append(cam.name)
        adjacency: Dict[str, List[str]] = {cam.name: [] for cam in self.cameras}
        for opening in self.openings:
            a, b = opening.between[0], opening.between[1]
            for cam_a in by_room.get(a, []):
                for cam_b in by_room.get(b, []):
                    if cam_b not in adjacency[cam_a]:
                        adjacency[cam_a].append(cam_b)
                    if cam_a not in adjacency[cam_b]:
                        adjacency[cam_b].append(cam_a)
        return {name: sorted(peers) for name, peers in adjacency.items()}

    def room_adjacency(self) -> Dict[str, List[str]]:
        adjacency: Dict[str, List[str]] = {room.id: [] for room in self.rooms}
        for opening in self.openings:
            a, b = opening.between[0], opening.between[1]
            if b not in adjacency[a]:
                adjacency[a].append(b)
            if a not in adjacency[b]:
                adjacency[b].append(a)
        return {room_id: sorted(peers) for room_id, peers in adjacency.items()}


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

    rooms: List[RoomSpec] = []
    for i, raw_room in enumerate(data["rooms"]):
        room_id = str(raw_room["id"])
        polygon = [_vec2(p, f"rooms[{i}].polygon vertex") for p in raw_room["polygon"]]
        _require(len(polygon) >= 3, f"room '{room_id}' needs at least 3 vertices")
        color = raw_room.get("floor_color", [0.6, 0.6, 0.6])
        _require(len(color) == 3, f"room '{room_id}' floor_color must have 3 components")
        rooms.append(
            RoomSpec(id=room_id, type=str(raw_room.get("type", "room")), polygon=polygon,
                     floor_color=[float(c) for c in color])
        )
    ids = [room.id for room in rooms]
    _require(len(set(ids)) == len(ids), "room ids must be unique")
    for room in rooms:
        _require(polygon_area(room.polygon) > 0.0,
                 f"room '{room.id}' polygon must be counter-clockwise with positive area")

    openings: List[OpeningSpec] = []
    for i, raw in enumerate(data.get("openings", [])):
        between = [str(r) for r in raw["between"]]
        _require(len(between) == 2, f"openings[{i}].between must list exactly two rooms")
        for room_id in between:
            _require(room_id in ids, f"openings[{i}] references unknown room '{room_id}'")
        width, height = float(raw["width"]), float(raw["height"])
        _require(width > 0.0 and height > 0.0, f"openings[{i}] needs positive width/height")
        _require(height <= wall_height + _EPS, f"openings[{i}].height exceeds wall_height")
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
        _require(sill + height <= wall_height + _EPS, f"windows[{i}] top edge exceeds wall_height")
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
                     lighting=lighting, front3d=raw_front3d, source_path=source_path, raw=data)
    validate_spec(spec)
    return spec


def load_spec(path: str) -> SceneSpec:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return parse_spec(data, source_path=path)


def validate_spec(spec: SceneSpec) -> None:
    """Geometric cross-checks that need the whole scene at once.

    Raises :class:`SpecError` on the first violation.
    """
    segments = spec.wall_segments()

    for i, opening in enumerate(spec.openings):
        label = f"openings[{i}] ({opening.kind} between {opening.between})"
        shared = [
            seg for seg in segments
            if all(room_id in seg.owners for room_id in opening.between) and seg.contains_point(opening.center)
        ]
        _require(bool(shared), f"{label}: center {opening.center} does not lie on a wall shared by both rooms")
        seg = shared[0]
        half = opening.width / 2.0
        along = (opening.center[0] - seg.p0[0]) * seg.direction[0] + (opening.center[1] - seg.p0[1]) * seg.direction[1]
        _require(-1e-6 <= along - half and along + half <= seg.length + 1e-6,
                 f"{label}: opening width {opening.width} does not fit into the shared wall span "
                 f"(span length {seg.length:.4f} m, centre offset {along:.4f} m)")
        _require(opening.height <= spec.wall_height + _EPS, f"{label}: height exceeds wall_height")

    for i, window in enumerate(spec.windows):
        label = f"windows[{i}] (room {window.room})"
        room = spec.room(window.room)
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
        seg = wall_segment_for_point(segments, middle, window.room)
        _require(seg is not None, f"{label}: window span is not on a wall of room '{window.room}'")
        length = math.dist(window.wall[0], window.wall[1])
        _require(length > 1e-3, f"{label}: window span is degenerate")
        _require(all(seg.contains_point(p, 1e-6) for p in window.wall),
                 f"{label}: window span crosses a wall junction or is longer than the wall span")
        _require(area > 0.0, f"{label}: room polygon has no area")

    for i, cam in enumerate(spec.cameras):
        label = f"cameras[{i}] ('{cam.name}')"
        room = spec.room(cam.room)
        _require(point_in_polygon(cam.pos, room.polygon),
                 f"{label}: position {cam.pos} is outside room '{cam.room}'")
        _require(0.0 < cam.pos[2] < spec.wall_height,
                 f"{label}: z={cam.pos[2]} must be in (0, wall_height={spec.wall_height})")

    for i, item in enumerate(spec.furniture):
        label = f"furniture[{i}] ('{item.asset}' in {item.room})"
        room = spec.room(item.room)
        _require(point_in_polygon(item.pos, room.polygon),
                 f"{label}: {item.pos} is outside room '{item.room}' (place furniture inside; "
                 f"the Blender side checks the bounding box against the walls)")
        if item.z is None and item.mount == "wall":
            raise SpecError(f"{label}: mount 'wall' needs an explicit z")
        if item.z is not None:
            _require(-0.01 <= item.z <= spec.wall_height,
                     f"{label}: z={item.z} is outside the room height")
        if item.asset == "builtin:rug":
            _require(item.size is not None and len(item.size) == 2 and min(item.size) > 0.2,
                     f"{label}: builtin:rug needs size [length, width] in metres")


def dump_manifest_spec(spec: SceneSpec) -> dict:
    """The original spec dict, with a stable key order for the manifest."""
    if spec.raw is not None:
        return spec.raw
    return {
        "unit": spec.unit,
        "wall_height": spec.wall_height,
        "wall_thickness": spec.wall_thickness,
        "rooms": [
            {"id": r.id, "type": r.type, "polygon": r.polygon, "floor_color": r.floor_color}
            for r in spec.rooms
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
