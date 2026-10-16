"""Autonomous apartment layout generator: seed in, scene spec out.

    python blender_agent/layout_gen.py --seed 0 --out spec.json \
        [--rooms 2..5] [--assets-index third_party/assets/polyhaven/index.json] \
        [--report layout_report.json]

The generator turns a random seed into a furnished, multi-room ``scene_spec``
that the existing pipeline can build, render and validate:

1. *shell*: 2..5 axis-aligned rooms laid out as one row (a "railroad" flat):
   rooms share the south edge, get depths between 3.65 m and 4.8 m, areas drawn
   from the realistic bands (living 18-30 m2, bedroom 12-20 m2, kitchen
   6-12 m2, bathroom 4-8 m2, study 8-14 m2) and a minimum side per type.  Every
   cut is vertical, so rooms abut on full-height shared walls, the floor plan is
   a guillotine partition of one rectangle and the room graph is a path.
2. *openings*: one 0.9 x 2.1 door per shared wall, all centred on a common door
   line near the south edge.
3. *windows*: 1-2 per habitable room on its own exterior wall spans
   (1.2-2.0 m wide, sill 0.9 m, height 1.4 m), kept clear of wall corners.
4. *furniture*: Poly Haven assets picked by category from the asset index (the
   API dimensions are millimetres and become the collision boxes), placed from
   per room-type templates (sofa against a wall facing the room, coffee table
   0.4-0.6 m in front of it, rug underneath, bookshelf on another wall, plant
   in a free corner, ceiling lamp at the centre, art above the sofa; bed head
   against a wall, nightstand next to it with a lamp on top; kitchen/bathroom
   stay empty or get a plant).  Every slot resamples its position up to
   ``SLOT_ATTEMPTS`` times and a failed required slot restarts the layout.
5. *cameras*: 1-2 per room, sampled on the room's **walkable grid** (the free
   cells of :func:`checks.circulation_report`, furniture footprints dilated)
   and picked by what they see: the share of the walkable floor and of the
   furniture in unobstructed view, the clearance to walls and furniture and the
   distance to the doorways.  Stands stay in the band around the door line and
   >= 0.6 m off the furniture, and a two-camera room gets two stands that look
   at it from different bearings.
6. *verification*: :mod:`blender_agent.checks` must pass, and the door depth
   spot check of ``tools/validate_blender_agent_p0.py`` is pre-computed with an
   analytic shell model (:class:`ShellModel`), so the rendered validation is not
   a coin flip.

Why a row, and why one door line
--------------------------------
``tools/validate_blender_agent_p0.py`` samples its door spot check at the column
of ``atan2(dy, dx)`` while the ERP convention (``spec.py``, verified against the
renders) maps a direction to ``atan2(-dy, dx)``.  The sampled ray is therefore
the mirror image of the door direction about the camera's +X axis, and in a
general 2-D plan the mirrored ray runs into the nearest wall: the check reports
a door/wall depth ratio near 1.0 and fails even for a perfectly good room (a
probe render confirmed this: measured 1.63/1.67 m against the 1.60/1.68 m this
model predicts).  With every door and every camera on one common Y the mirror
maps the geometry onto itself, the sample hits the real sight line through the
doorway, and the measured ratio matches the forecast (2-3.5x, exactly what the
hand-tuned P1 scene shows).  ``keep_room_open``-style rules (a clear sight
corridor, tall furniture out of it) are what makes that line stay clear.

The forecast is only worth trusting because it *measures* the way the validator
measures: the same column arithmetic, the same pixel rows (``_sample_row``) and
the same 5x5 window median (:meth:`ShellModel._window_median`).  Both details
matter at long range -- half a row of rounding is 4 mm at a door 8 m away, and
the width of that margin decides whether the wall ray grazes over a doorway
lintel (a 2.2x ratio) or slips under it (1.4x, which fails the render check).

Outputs ``scene_spec.json`` (strictly the schema ``blender_agent/spec.py``
accepts) plus ``layout_report.json`` (room graph, furniture list, per-slot
provenance, check results and the door forecast).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import checks as checks_mod  # noqa: E402
from blender_agent import spec as spec_mod  # noqa: E402

DEFAULT_ASSETS_INDEX = checks_mod.DEFAULT_ASSETS_INDEX
DEFAULT_REPORT_NAME = "layout_report.json"
PANO_W = spec_mod.PANO_WIDTH
PANO_H = spec_mod.PANO_HEIGHT

WALL_HEIGHT = 2.7
WALL_THICKNESS = 0.12
DOOR_W = 0.9
DOOR_H = 2.1
WINDOW_SILL = 0.9
WINDOW_H = 1.4
WINDOW_W_RANGE = (1.2, 2.0)
WINDOW_MARGIN_M = 0.35

HABITABLE = ("living_room", "bedroom", "study", "kitchen")
ROOM_AREA_M2 = {
    "living_room": (18.0, 30.0),
    "bedroom": (12.0, 20.0),
    "kitchen": (6.0, 12.0),
    "bathroom": (4.0, 8.0),
    "study": (8.0, 14.0),
}
ROOM_MIN_DIM_M = {"living_room": 3.2, "bedroom": 2.7, "kitchen": 2.0, "bathroom": 1.7, "study": 2.4}
ROOM_FLOOR_COLOR = {
    "living_room": [0.55, 0.38, 0.22],
    "bedroom": [0.60, 0.50, 0.40],
    "kitchen": [0.62, 0.61, 0.58],
    "bathroom": [0.72, 0.73, 0.74],
    "study": [0.50, 0.36, 0.24],
}
ROOM_FLOOR_COLOR_DEFAULT = [0.58, 0.50, 0.42]

GRID_STEP_M = 0.05
SNAP_M = 0.05
MIN_SHARED_WALL_M = 1.4
ROOM_DEPTH_MIN_M = 3.65
ROOM_DEPTH_MAX_M = 4.80
ROOM_AREA_HEADROOM_M = 0.30
CORRIDOR_HALF_M = 0.35
CORRIDOR_MAX_H_M = 1.05
ROOM_OPEN_FRACTION = 0.85
MAX_LAYOUT_ATTEMPTS = 80
SLOT_ATTEMPTS = 8
CAMERA_INSET_M = 0.40
CAMERA_GRID_M = 0.25
CAMERA_ROW_STEPS = 3
CAMERA_BAND_M = 0.35
CAMERA_Z_M = 1.5
CAMERA_MIN_GAP_TO_FURNITURE_M = 0.60
CAMERA_DOOR_GAP_M = 0.60
CAMERA_RATIO_WEIGHT = 0.35
DOOR_FORECAST_MIN_RATIO = 1.65
# camera sampling (step 5): stands are drawn from the room's walkable grid and
# scored by what they can see -- see ``place_cameras``
CAMERA_VIS_GRID_M = 0.60
CAMERA_TARGET_Z_M = 0.45
CAMERA_VIS_TOLERANCE_M = 0.05
CAMERA_COVERAGE_WEIGHT = 2.0
CAMERA_FURNITURE_WEIGHT = 0.8
CAMERA_WALL_WEIGHT = 0.6
CAMERA_CLEARANCE_WEIGHT = 0.4
CAMERA_DOOR_WEIGHT = 0.6
CAMERA_WALL_COMFORT_M = 1.20
CAMERA_CLEARANCE_COMFORT_M = 1.20
CAMERA_SECOND_MIN_GAP_M = 1.20
CAMERA_SECOND_MIN_ANGLE_DEG = 45.0
CAMERA_SECOND_GAP_WEIGHT = 0.35
CAMERA_SECOND_ANGLE_WEIGHT = 0.8

STYLE_DEFAULTS = {
    "sun_elevation_deg": 35,
    "sun_azimuth_deg": 210,
    "sky_strength": 1.0,
    "sun_energy": 3.0,
    "view_transform": "AgX",
    "exposure": 1.4,
}
MATERIAL_DEFAULTS = {
    "floor": "ph:oak_wood_planks",
    "wall": "ph:beige_wall_001",
    "rug": "ph:poly_wool_herringbone",
    "floor_tile_m": 1.2,
    "wall_tile_m": 3.0,
    "rug_tile_m": 0.9,
    "rug_tint": [0.40, 0.36, 0.32],
    "rug_tint_mix": 0.55,
    "normal_strength": {"floor": 0.45, "wall": 0.7, "rug": 0.3},
}
DETAIL_DEFAULTS = {"skirting": True, "casings": True, "bevels": True, "glass": True}
LIGHTING_DEFAULTS = {"window_portal_w": 42.0, "lamp_w": 8.0, "portal_margin": 0.02}

# --------------------------------------------------------------------------- #
# slot definitions: asset picked by category, fallbacks by measured name
# --------------------------------------------------------------------------- #
ASSET_SLOTS: Dict[str, dict] = {
    "sofa": {"categories": ("furniture", "seating"), "names": ("sofa_02",),
             "dims": lambda d: 1.4 <= d[0] <= 2.4 and d[1] <= 1.1},
    "armchair": {"categories": ("furniture", "seating"), "names": ("modern_arm_chair_01",),
                 "dims": lambda d: 0.6 <= d[0] <= 1.0 and d[1] <= 1.2},
    "coffee_table": {"categories": ("furniture", "table"), "names": ("modern_coffee_table_01",),
                     "dims": lambda d: 0.8 <= d[0] <= 1.6 and d[2] <= 0.6},
    "bookshelf": {"categories": ("furniture", "shelves"), "names": ("wooden_bookshelf_worn",),
                  "dims": lambda d: 0.8 <= d[0] <= 2.0 and d[1] <= 0.8},
    "bed": {"categories": ("furniture", "bed"), "names": ("GothicBed_01", "old_bed_frame"),
            "dims": lambda d: 1.2 <= d[0] <= 1.9 and 1.7 <= d[1] <= 2.4},
    "nightstand": {"categories": ("furniture", "table"), "names": ("painted_wooden_nightstand",),
                   "dims": lambda d: 0.3 <= d[0] <= 0.7 and d[1] <= 0.7 and d[2] <= 0.8},
    "plant_tall": {"categories": ("decorative", "plants"), "names": ("potted_plant_01",),
                   "dims": lambda d: 0.3 <= d[0] <= 1.0 and d[2] >= 1.0},
    "plant_small": {"categories": ("decorative", "plants"), "names": ("potted_plant_02",),
                    "dims": lambda d: 0.3 <= d[0] <= 1.0},
    "ceiling_lamp": {"categories": ("lighting",), "names": ("modern_ceiling_lamp_01",),
                     "dims": lambda d: d[2] <= 1.2 and d[0] <= 0.8},
    "art": {"categories": ("decorative", "wall decoration"), "names": ("hanging_picture_frame_01",),
            "dims": lambda d: 0.3 <= d[0] <= 1.2 and d[2] <= 1.2},
    "basket": {"categories": ("containers",), "names": ("wicker_basket_01",),
               "dims": lambda d: d[2] <= 0.5},
}
RUG_ASSET = "builtin:rug"
LAMP_ASSET = "builtin:table_lamp"

BACK_GAP_M = WALL_THICKNESS / 2.0 + 0.02      # wall centre line -> furniture back
ART_GAP_M = WALL_THICKNESS / 2.0 + 0.01       # wall centre line -> frame centre
WALL_ROT_DEG = {"N": 0.0, "S": 180.0, "W": 90.0, "E": 270.0}
WALL_FACING = {"N": (0.0, -1.0), "S": (0.0, 1.0), "W": (1.0, 0.0), "E": (-1.0, 0.0)}


class LayoutError(RuntimeError):
    """Raised when no valid layout could be generated for a seed."""


class _AttemptFailed(Exception):
    """Internal: restart the whole layout with a fresh random stream."""


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #
def _round(value: float, step: float = SNAP_M) -> float:
    return round(value / step) * step


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rotated_half_extents(dims: Sequence[float], rot_z_deg: float) -> Tuple[float, float]:
    angle = math.radians(float(rot_z_deg))
    cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
    return (0.5 * (dims[0] * cos_a + dims[1] * sin_a), 0.5 * (dims[0] * sin_a + dims[1] * cos_a))


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    def polygon(self) -> List[List[float]]:
        return [[self.x0, self.y0], [self.x1, self.y0], [self.x1, self.y1], [self.x0, self.y1]]


@dataclass
class Wall:
    """One side of a room, in world axes."""

    side: str
    line: float
    lo: float
    hi: float

    @property
    def length(self) -> float:
        return self.hi - self.lo

    @property
    def rot_deg(self) -> float:
        return WALL_ROT_DEG[self.side]

    @property
    def facing(self) -> Tuple[float, float]:
        return WALL_FACING[self.side]

    def contains(self, along: float, margin: float = 0.0) -> bool:
        return self.lo + margin <= along <= self.hi - margin


def room_walls(rect: Rect) -> Dict[str, Wall]:
    return {
        "S": Wall("S", rect.y0, rect.x0, rect.x1),
        "N": Wall("N", rect.y1, rect.x0, rect.x1),
        "W": Wall("W", rect.x0, rect.y0, rect.y1),
        "E": Wall("E", rect.x1, rect.y0, rect.y1),
    }


@dataclass
class RoomPlan:
    room_id: str
    room_type: str
    rect: Rect
    target_area: float
    width: float
    depth: float

    @property
    def walls(self) -> Dict[str, Wall]:
        return room_walls(self.rect)


@dataclass
class DoorPlan:
    a: str
    b: str
    center: Tuple[float, float]
    span: Tuple[float, float]
    wall: Wall
    direction: Tuple[float, float]

    def key(self) -> str:
        return f"{self.a}|{self.b}"


@dataclass
class Item:
    name: str
    asset: str
    room: str
    pos: List[float]
    rot_z_deg: float
    z: Optional[float]
    mount: str
    size: Optional[List[float]]
    dims: Tuple[float, float, float]
    placement: str
    attached_to: Optional[str] = None
    slot: str = ""

    @property
    def aabb(self) -> Tuple[float, float, float, float]:
        half_x, half_y = rotated_half_extents(self.dims, self.rot_z_deg)
        return (self.pos[0] - half_x, self.pos[1] - half_y, self.pos[0] + half_x, self.pos[1] + half_y)

    @property
    def z_range(self) -> Tuple[float, float]:
        return (self.z or 0.0, (self.z or 0.0) + self.dims[2])

    def to_json(self) -> dict:
        entry: Dict[str, object] = {"asset": self.asset, "room": self.room,
                                    "pos": [round(self.pos[0], 4), round(self.pos[1], 4)],
                                    "rot_z_deg": self.rot_z_deg, "name": self.name}
        if self.z is not None:
            entry["z"] = round(self.z, 4)
        if self.mount != "floor":
            entry["mount"] = self.mount
        if self.size is not None:
            entry["size"] = [round(v, 3) for v in self.size]
        if self.attached_to:
            entry["attached_to"] = self.attached_to
        if self.placement:
            entry["placement"] = self.placement
        return entry


# --------------------------------------------------------------------------- #
# step 1: rooms
# --------------------------------------------------------------------------- #
def choose_room_types(rng: random.Random, count: int) -> List[str]:
    """Living room always, at least one bedroom, then kitchen/study/bathroom picks."""
    types = ["living_room"]
    if count >= 2:
        types.append("bedroom")
    pool = ["bedroom", "bedroom", "kitchen", "study", "bathroom", "bedroom"]
    used: Dict[str, int] = {"bedroom": 1 if count >= 2 else 0, "kitchen": 0, "bathroom": 0, "study": 0}
    while len(types) < count:
        pick = rng.choice(pool)
        if pick == "bedroom" and used["bedroom"] >= 2:
            continue
        if pick in ("kitchen", "bathroom") and used[pick] >= 1:
            continue
        used[pick] = used.get(pick, 0) + 1
        types.append(pick)
    rng.shuffle(types)
    return types


def plan_rooms(rng: random.Random, types: Sequence[str]) -> List[RoomPlan]:
    """Lay the rooms out as a single row (a "railroad" flat) along +X.

    Rooms share one south edge (y = 0) and may have different depths, so the
    plan is a guillotine partition of one rectangle: every cut is vertical, the
    rooms abut on full-height shared walls and the room graph is a path.

    The row is not just a style choice.  ``tools/validate_blender_agent_p0.py``
    samples its door spot check at the column of ``atan2(dy, dx)`` while the ERP
    convention of ``spec.py`` maps a direction to ``atan2(-dy, dx)``: the sampled
    ray is the mirror image of the door direction about the camera's +X axis.
    With every door and every camera on one common Y the mirror is the identity
    and the check measures the real sight line through the door; in a general
    2-D plan the mirrored ray points into a wall and the check fails no matter
    how the furniture is arranged (see ``ShellModel`` and the layout report).
    """
    plans: List[RoomPlan] = []
    depth_base = rng.uniform(3.55, 4.25)
    for room_type in types:
        low, high = ROOM_AREA_M2[room_type]
        min_dim = ROOM_MIN_DIM_M[room_type]
        depth = min(depth_base + rng.uniform(-0.30, 0.45),
                    (high - ROOM_AREA_HEADROOM_M) / min_dim - 0.15, ROOM_DEPTH_MAX_M)
        depth = max(depth, ROOM_DEPTH_MIN_M)
        area_low = max(low, min_dim * depth)
        if area_low > high - 1e-6 or depth < min_dim - 1e-9:
            raise _AttemptFailed(f"{room_type} does not fit a {depth:.2f} m deep row")
        area_high = high - ROOM_AREA_HEADROOM_M
        area = rng.uniform(area_low + 0.2 * (area_high - area_low), area_high)
        if area_low + 0.05 > area_high:
            raise _AttemptFailed(f"{room_type} area floor {area_low:.2f} leaves no spread below {high}")
        plans.append(RoomPlan(room_id="", room_type=room_type, rect=Rect(0, 0, 0, 0),
                              target_area=area, width=area / depth, depth=depth))
    rng.shuffle(plans)
    x = 0.0
    for plan in plans:
        x1 = _round(x + plan.width)
        plan.rect = Rect(x, 0.0, x1, _round(plan.depth, 0.01))
        x = x1
    if x < 6.0:
        raise _AttemptFailed("row too short")
    ids: Dict[str, int] = {}
    for plan in plans:
        base = {"living_room": "living", "bedroom": "bedroom", "kitchen": "kitchen",
                "bathroom": "bathroom", "study": "study"}[plan.room_type]
        ids[base] = ids.get(base, 0) + 1
        plan.room_id = base if ids[base] == 1 else f"{base}_{ids[base]}"
    for plan in plans:
        low, high = ROOM_AREA_M2[plan.room_type]
        if plan.rect.w < ROOM_MIN_DIM_M[plan.room_type] - 1e-9 or \
                plan.rect.h < ROOM_MIN_DIM_M[plan.room_type] - 1e-9:
            raise _AttemptFailed(f"room {plan.room_type} too small "
                                 f"({plan.rect.w:.2f}x{plan.rect.h:.2f} m)")
        if not (low - 1e-9 <= plan.rect.area <= high + 1e-9):
            raise _AttemptFailed(f"room {plan.room_type} area {plan.rect.area:.1f} m2 outside "
                                 f"[{low}, {high}]")
    return plans


# --------------------------------------------------------------------------- #
# step 2: doors
# --------------------------------------------------------------------------- #
def shared_wall_span(a: Rect, b: Rect) -> Optional[Tuple[Wall, Tuple[float, float]]]:
    """``(wall of ``a``, (lo, hi) overlap)`` when the two rectangles touch along a wall."""
    tol = 1e-6
    if abs(a.x1 - b.x0) < tol:
        lo, hi = max(a.y0, b.y0), min(a.y1, b.y1)
        if hi - lo > tol:
            return Wall("E", a.x1, lo, hi), (lo, hi)
    if abs(a.x0 - b.x1) < tol:
        lo, hi = max(a.y0, b.y0), min(a.y1, b.y1)
        if hi - lo > tol:
            return Wall("W", a.x0, lo, hi), (lo, hi)
    if abs(a.y1 - b.y0) < tol:
        lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
        if hi - lo > tol:
            return Wall("N", a.y1, lo, hi), (lo, hi)
    if abs(a.y0 - b.y1) < tol:
        lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
        if hi - lo > tol:
            return Wall("S", a.y0, lo, hi), (lo, hi)
    return None


def plan_doors(rng: random.Random, plans: Sequence[RoomPlan]) -> List[DoorPlan]:
    """One door per shared wall of the row, all centred on a common Y line.

    The row's rooms are neighbours in X, so the shared-wall graph is a path and
    a spanning tree is simply "every shared wall gets a door".  Every door sits
    at ``y_star`` (plus a little jitter) so that a camera anywhere in the row can
    see straight down the row through the doorways -- the sight line the depth
    validation measures.
    """
    ordered = sorted(plans, key=lambda plan: plan.rect.x0)
    pairs: List[Tuple[int, int, Wall, float, float]] = []
    for index in range(len(ordered) - 1):
        shared = shared_wall_span(ordered[index].rect, ordered[index + 1].rect)
        if shared is None:
            continue
        wall, (lo, hi) = shared
        if hi - lo < MIN_SHARED_WALL_M:
            continue
        pairs.append((index, index + 1, wall, lo, hi))
    if len(pairs) != len(ordered) - 1:
        raise _AttemptFailed("the row is not a chain of rooms")
    min_depth = min(plan.rect.h for plan in ordered)
    if min_depth < ROOM_DEPTH_MIN_M - 1e-9:
        raise _AttemptFailed("row too shallow for a shared door line")
    # Door line near the south wall: it keeps the enfilade corridor (and the
    # 0.85 m door approach) out of the way of the tall furniture that needs the
    # north half of a room (a 2.04 m bed against a side wall).
    y_star = rng.uniform(1.05, 1.20)
    y_star = _clamp(y_star, 0.9, min_depth - 0.9)
    doors: List[DoorPlan] = []
    for index, _next, wall, lo, hi in pairs:
        margin = DOOR_W / 2.0 + 0.20
        if hi - lo < DOOR_W + 2 * 0.20:
            raise _AttemptFailed("shared wall too short for a door")
        low, high = max(lo + margin, y_star - 0.12), min(hi - margin, y_star + 0.12)
        if low > high:
            raise _AttemptFailed("door line does not fit the shared wall")
        center_along = _round(rng.uniform(low, high), 0.01)
        doors.append(DoorPlan(a=ordered[index].room_id, b=ordered[index + 1].room_id,
                              center=(wall.line, center_along),
                              span=(center_along - DOOR_W / 2.0, center_along + DOOR_W / 2.0),
                              wall=wall, direction=(0.0, 1.0)))
    return doors


# --------------------------------------------------------------------------- #
# step 3: windows
# --------------------------------------------------------------------------- #
def exterior_spans(plans: Sequence[RoomPlan]):
    """Exterior wall spans per room id: a span is exterior when a single room owns it."""
    room_specs = [spec_mod.RoomSpec(id=plan.room_id, type=plan.room_type, polygon=plan.rect.polygon())
                  for plan in plans]
    segments = spec_mod.build_wall_segments(room_specs)
    spans = {plan.room_id: [] for plan in plans}
    for segment in segments:
        if len(segment.owners) != 1:
            continue
        spans[segment.owners[0]].append(segment)
    return spans


def plan_windows(rng: random.Random, plans: Sequence[RoomPlan]) -> Dict[str, List[dict]]:
    """One window per exterior span, 1-2 per habitable room (bathrooms rarely)."""
    spans = exterior_spans(plans)
    windows: Dict[str, List[dict]] = {plan.room_id: [] for plan in plans}
    for plan in plans:
        candidates = []
        for segment in spans[plan.room_id]:
            (ax, ay), (bx, by) = segment.p0, segment.p1
            length = math.hypot(bx - ax, by - ay)
            if length >= WINDOW_W_RANGE[0] + 2 * WINDOW_MARGIN_M:
                candidates.append((length, (ax, ay), (bx, by)))
        if not candidates:
            continue
        if plan.room_type in HABITABLE:
            want = 2 if len(candidates) > 1 and rng.random() < 0.4 else 1
        elif plan.room_type == "bathroom":
            want = 1 if rng.random() < 0.25 else 0
        else:
            want = 0
        rng.shuffle(candidates)
        candidates.sort(key=lambda entry: -entry[0])
        for _, start, end in candidates:
            if len(windows[plan.room_id]) >= want:
                break
            length = math.dist(start, end)
            ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
            usable = length - 2 * WINDOW_MARGIN_M
            width = _clamp(rng.uniform(*WINDOW_W_RANGE), WINDOW_W_RANGE[0],
                           min(WINDOW_W_RANGE[1], usable))
            low = WINDOW_MARGIN_M + width / 2.0
            high = length - WINDOW_MARGIN_M - width / 2.0
            offset = _round(rng.uniform(low, high), 0.05)
            offset = _clamp(offset, low, high)
            half = width / 2.0
            wall_a = [round(start[0] + ux * (offset - half), 4), round(start[1] + uy * (offset - half), 4)]
            wall_b = [round(start[0] + ux * (offset + half), 4), round(start[1] + uy * (offset + half), 4)]
            if math.dist(wall_a, wall_b) < 1e-3:
                continue
            windows[plan.room_id].append({"room": plan.room_id, "wall": [wall_a, wall_b],
                                          "sill": WINDOW_SILL, "height": WINDOW_H})
    return windows


# --------------------------------------------------------------------------- #
# step 4: furniture
# --------------------------------------------------------------------------- #
def door_rects_for(rect: Rect, doors: Sequence[DoorPlan]) -> List[Tuple[float, float, float, float]]:
    """Door approach rectangles on this room's side of every door it owns.

    Same numbers as ``checks.DOOR_ZONE_DEPTH_M`` / ``DOOR_ZONE_EXTRA_W_M`` so
    the generator and the checker cannot drift apart.
    """
    rects: List[Tuple[float, float, float, float]] = []
    for door in doors:
        half_w = DOOR_W / 2.0 + checks_mod.DOOR_ZONE_EXTRA_W_M / 2.0
        depth = checks_mod.DOOR_ZONE_DEPTH_M + WALL_THICKNESS / 2.0
        if door.wall.side in ("N", "S"):
            x_c = door.center[0]
            if door.wall.line >= rect.center[1]:
                rects.append((x_c - half_w, door.wall.line - depth, x_c + half_w, door.wall.line + 0.2))
            else:
                rects.append((x_c - half_w, door.wall.line - 0.2, x_c + half_w, door.wall.line + depth))
        else:
            y_c = door.center[1]
            if door.wall.line >= rect.center[0]:
                rects.append((door.wall.line - depth, y_c - half_w, door.wall.line + 0.2, y_c + half_w))
            else:
                rects.append((door.wall.line - 0.2, y_c - half_w, door.wall.line + depth, y_c + half_w))
    return rects


class Placer:
    """Rejection sampling for one room: keeps the layout contract of checks.py."""

    def __init__(self, room: RoomPlan, doors: Sequence[DoorPlan], windows: Sequence[dict],
                 index: Dict[str, dict], rng: random.Random, placed: List[Item], report: dict):
        self.room = room
        self.rect = room.rect
        self.doors = [door for door in doors if room.room_id in (door.a, door.b)]
        self.windows = list(windows)
        self.index = index
        self.rng = rng
        self.placed = placed
        self.report = report
        self.walls = room.walls
        self.rejections: Dict[str, int] = {}

    # -- helpers --------------------------------------------------------- #
    def pick_asset(self, slot: str) -> Optional[str]:
        spec_slot = ASSET_SLOTS[slot]

        def ref_of(name: str) -> str:
            entry = self.index.get(name) or {}
            return entry.get("ref") or f"ph:{name}"

        # Sample uniformly from every library entry that fits the slot, so each
        # house draws different furniture; the named list is only the fallback
        # when the category pool is empty.
        matches = []
        for name in sorted(self.index):
            entry = self.index[name]
            dims = entry.get("dims_m")
            if not dims:
                continue
            categories = set(entry.get("categories") or ())
            if not all(category in categories for category in spec_slot["categories"]):
                continue
            if spec_slot["dims"](dims):
                matches.append(name)
        if matches:
            return ref_of(self.rng.choice(matches))
        for name in spec_slot["names"]:
            entry = self.index.get(name)
            if entry and entry.get("dims_m") and spec_slot["dims"](entry["dims_m"]):
                return ref_of(name)
        return None

    def dims_of(self, asset: str) -> Optional[Tuple[float, float, float]]:
        return checks_mod.item_dims(asset, self.index)

    def door_rects(self) -> List[Tuple[float, float, float, float]]:
        """Door approach rectangles of this room, using the shared contract."""
        return door_rects_for(self.rect, self.doors)

    def wall_has_window(self, wall: Wall) -> bool:
        for window in self.windows:
            (ax, ay), (bx, by) = window["wall"]
            if wall.side in ("N", "S"):
                if abs(ay - wall.line) < 1e-6 and abs(by - wall.line) < 1e-6:
                    return True
            else:
                if abs(ax - wall.line) < 1e-6 and abs(bx - wall.line) < 1e-6:
                    return True
        return False

    def inside(self, aabb, gap: float) -> bool:
        return (aabb[0] >= self.rect.x0 + gap - 1e-9 and aabb[2] <= self.rect.x1 - gap + 1e-9
                and aabb[1] >= self.rect.y0 + gap - 1e-9 and aabb[3] <= self.rect.y1 - gap + 1e-9)

    def corridor_band(self) -> Optional[Tuple[float, float]]:
        """Y band the door sight lines travel through (see ``fits``)."""
        if not self.doors:
            return None
        low = min(door.center[1] for door in self.doors) - CORRIDOR_HALF_M
        high = max(door.center[1] for door in self.doors) + CORRIDOR_HALF_M
        return (low, high)

    def fits(self, item: Item, gap: float, keep_doors: bool = True) -> bool:
        aabb = item.aabb
        if not self.inside(aabb, gap):
            self.reject("outside_room")
            return False
        if item.dims[2] > CORRIDOR_MAX_H_M:
            band = self.corridor_band()
            if band is not None and aabb[1] < band[1] - 1e-9 and aabb[3] > band[0] + 1e-9:
                self.reject("sight_corridor")
                return False
        if keep_doors:
            for rect in self.door_rects():
                if checks_mod.rect_distance(aabb, rect) < checks_mod.DOOR_CLEARANCE_M + 0.05:
                    self.reject("door_zone")
                    return False
        z_range = item.z_range
        for other in self.placed:
            if z_range[0] > other.z_range[1] - 1e-3 or other.z_range[0] > z_range[1] - 1e-3:
                continue
            if (aabb[2] <= other.aabb[0] + 1e-3 or other.aabb[2] <= aabb[0] + 1e-3
                    or aabb[3] <= other.aabb[1] + 1e-3 or other.aabb[3] <= aabb[1] + 1e-3):
                continue
            if item.placement == checks_mod.FLOOR_FLAT or other.placement == checks_mod.FLOOR_FLAT:
                continue
            if item.placement == checks_mod.CEILING or other.placement == checks_mod.CEILING:
                continue
            if item.attached_to == other.name:
                continue
            self.reject(f"collision:{item.slot}x{other.slot}")
            return False
        return True

    def keeps_room_open(self, item: Item) -> bool:
        """Reject a placement that would fence off part of the room.

        Runs a coarse circulation pass (``checks.room_connectivity``) over the
        room with this item added; a layout where the furniture chain cuts the
        room in two fails the real 5 cm check later, so it is cheaper to reject
        it here.
        """
        if not item.aabb or item.placement in (checks_mod.FLOOR_FLAT, checks_mod.CEILING, checks_mod.MOUNTED):
            return True
        area = (item.aabb[2] - item.aabb[0]) * (item.aabb[3] - item.aabb[1])
        if area < 0.5 or not _blocks_camera(item):
            return True
        boxes = [checks_mod.ItemBox(name=other.name, asset=other.asset, room=other.room, aabb=other.aabb,
                                    z=other.z_range, dims=other.dims, placement=other.placement,
                                    rot_z_deg=other.rot_z_deg, rotation_on_grid=True)
                 for other in self.placed] + \
                [checks_mod.ItemBox(name=item.name, asset=item.asset, room=item.room, aabb=item.aabb,
                                    z=item.z_range, dims=item.dims, placement=item.placement,
                                    rot_z_deg=item.rot_z_deg, rotation_on_grid=True)]
        starts = [door.center for door in self.doors] or [(self.rect.center[0], self.rect.center[1])]
        stats = checks_mod.room_connectivity(self.rect.polygon(), boxes, starts,
                                            wall_thickness=WALL_THICKNESS, grid_m=0.20)
        if stats["fraction"] < ROOM_OPEN_FRACTION:
            self.reject("seals_room")
            return False
        return True

    def reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def add(self, item: Item) -> Item:
        self.placed.append(item)
        self.report.setdefault("slots", []).append({
            "slot": item.slot, "name": item.name, "asset": item.asset, "room": item.room,
            "placement": item.placement, "pos": [round(item.pos[0], 3), round(item.pos[1], 3)],
            "rot_z_deg": item.rot_z_deg, "size_m": [round(v, 3) for v in item.dims],
        })
        return item

    def make(self, slot: str, name: str, asset: str, pos, rot, placement: str,
             z: Optional[float] = None, size: Optional[Sequence[float]] = None,
             attached_to: Optional[str] = None) -> Optional[Item]:
        dims = checks_mod.item_dims(asset, self.index, size)
        if dims is None:
            return None
        return Item(name=name, asset=asset, room=self.room.room_id, pos=[float(pos[0]), float(pos[1])],
                    rot_z_deg=float(rot), z=z, mount="ceiling" if placement == checks_mod.CEILING else "floor",
                    size=[float(v) for v in size] if size else None, dims=dims,
                    placement=placement, attached_to=attached_to, slot=slot)

    # -- against a wall --------------------------------------------------- #
    def against_wall(self, slot: str, name: str, asset: str, wall: Wall, along: float,
                     placement: str = checks_mod.WALL_HUG, z: Optional[float] = None,
                     gap: float = BACK_GAP_M, size=None, attached_to=None) -> Optional[Item]:
        dims = checks_mod.item_dims(asset, self.index, size)
        if dims is None:
            return None
        half_x, half_y = rotated_half_extents(dims, wall.rot_deg)
        if wall.side == "N":
            pos = (along, wall.line - gap - half_y)
        elif wall.side == "S":
            pos = (along, wall.line + gap + half_y)
        elif wall.side == "W":
            pos = (wall.line + gap + half_x, along)
        else:
            pos = (wall.line - gap - half_x, along)
        item = self.make(slot, name, asset, pos, wall.rot_deg, placement, z=z, size=size,
                         attached_to=attached_to)
        if item is None:
            return None
        return item

    def along_limits(self, wall: Wall, half_along: float, margin: float = 0.10) -> Optional[Tuple[float, float]]:
        low, high = wall.lo + half_along + margin, wall.hi - half_along - margin
        if low > high + 1e-9:
            return None
        return low, high

    def along_intervals(self, wall: Wall, extent_along: float, extent_across: float,
                        height: float, margin: float = 0.10) -> List[Tuple[float, float]]:
        """Positions along ``wall`` where the item clears the door sight corridor.

        A tall item (taller than the eye-height ray that the door check casts)
        must stay out of the Y band between the doors, otherwise it cuts the
        enfilade line: on a side wall that is a condition on the position along
        the wall, on a north/south wall it depends only on the wall itself.
        """
        limits = self.along_limits(wall, extent_along / 2.0, margin)
        if limits is None:
            return []
        low, high = limits
        band = self.corridor_band()
        if band is None or height <= CORRIDOR_MAX_H_M:
            return [(low, high)]
        if wall.side in ("N", "S"):
            inner = wall.line - BACK_GAP_M - extent_across if wall.side == "N" else wall.line + BACK_GAP_M
            if inner < band[1] and inner + extent_across > band[0]:
                return []
            return [(low, high)]
        forbidden = (band[0] - extent_along / 2.0, band[1] + extent_along / 2.0)
        out = []
        if low < forbidden[0] - 1e-9:
            out.append((low, min(high, forbidden[0])))
        if high > forbidden[1] + 1e-9:
            out.append((max(low, forbidden[1]), high))
        return [(a, b) for a, b in out if b - a > 1e-9]

    def sample_along(self, intervals: Sequence[Tuple[float, float]]) -> float:
        """Pick a position from the intervals, weighted by interval length."""
        weights = [high - low for low, high in intervals]
        total = sum(weights)
        threshold = self.rng.uniform(0.0, total)
        running = 0.0
        for (low, high), weight in zip(intervals, weights):
            running += weight
            if threshold <= running or (low, high) == intervals[-1]:
                return _round(self.rng.uniform(low, high), 0.05)
        return intervals[-1][0]


def _place_on_wall(placer: Placer, slot: str, name: str, asset: str, walls: List[Wall],
                   gap: float = BACK_GAP_M, placement: str = checks_mod.WALL_HUG,
                   z: Optional[float] = None,
                   avoid: Sequence[Tuple[float, float, float, float]] = (), size=None,
                   attached_to=None, window_penalty: bool = True) -> Optional[Item]:
    """Try each wall (and positions along it) until the item fits."""
    dims = checks_mod.item_dims(asset, placer.index, size)
    if dims is None:
        return None
    scored = []
    for wall in walls:
        across, along_extent = dims[1], dims[0]
        intervals = placer.along_intervals(wall, along_extent, across, dims[2])
        if not intervals:
            continue
        score = sum(high - low for low, high in intervals)
        if window_penalty and placer.wall_has_window(wall):
            score -= 1.5
        scored.append((score, wall, intervals))
    scored.sort(key=lambda entry: -entry[0])
    for score, wall, intervals in scored[:3]:
        for attempt in range(SLOT_ATTEMPTS):
            along = placer.sample_along(intervals)
            item = placer.against_wall(slot, name, asset, wall, along, placement=placement, z=z,
                                       gap=gap, size=size, attached_to=attached_to)
            if item is None:
                return None
            if any(checks_mod.rect_distance(item.aabb, rect) < 0.10 for rect in avoid):
                continue
            if placer.fits(item, checks_mod.WALL_GAP_MIN_M) and placer.keeps_room_open(item):
                return placer.add(item)
    return None


def _place_free(placer: Placer, slot: str, name: str, asset: str, candidates,
                placement: str = checks_mod.FREE_STAND, size=None, attempts: int = 26,
                rot_fn=None) -> Optional[Item]:
    """Grid/random search for a free standing spot (>= 0.6 m off every wall)."""
    tried = list(candidates)
    placer.rng.shuffle(tried)
    for pos in tried[:attempts]:
        item = placer.make(slot, name, asset, pos, 0.0, placement, size=size)
        if item is None:
            return None
        rot_choices = [0.0, 90.0, 180.0, 270.0]
        placer.rng.shuffle(rot_choices)
        if rot_fn is not None:
            rot_choices = [rot_fn(pos)] + [rot for rot in rot_choices if rot != rot_fn(pos)]
        for rot in rot_choices:
            item.rot_z_deg = rot
            if placer.fits(item, checks_mod.WALL_GAP_FREE_MIN_M) and placer.keeps_room_open(item):
                return placer.add(item)
    return None


def _free_candidates(placer: Placer, dims, inset: float = checks_mod.WALL_GAP_FREE_MIN_M,
                     step: float = 0.25) -> List[Tuple[float, float]]:
    half_x, half_y = dims[0] / 2.0, dims[1] / 2.0
    out = []
    x = placer.rect.x0 + inset + half_x
    while x <= placer.rect.x1 - inset - half_x + 1e-9:
        y = placer.rect.y0 + inset + half_y
        while y <= placer.rect.y1 - inset - half_y + 1e-9:
            out.append((x, y))
            y += step
        x += step
    return out


def furnish_living(placer: Placer, want_armchair: bool) -> None:
    rng = placer.rng
    rect = placer.rect
    sofa_asset = placer.pick_asset("sofa")
    if sofa_asset is None:
        raise _AttemptFailed("no sofa asset available")
    sofa_dims = placer.dims_of(sofa_asset)
    walls = sorted(placer.walls.values(), key=lambda wall: -wall.length)
    sofa = None
    for wall in walls[:3]:
        across, along_extent = sofa_dims[1], sofa_dims[0]
        intervals = placer.along_intervals(wall, along_extent, across, sofa_dims[2], margin=0.25)
        if not intervals:
            continue
        for _ in range(SLOT_ATTEMPTS):
            along = placer.sample_along(intervals)
            item = placer.against_wall("sofa", "sofa", sofa_asset, wall, along)
            if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M) \
                    and placer.keeps_room_open(item):
                sofa = placer.add(item)
                break
        if sofa is not None:
            break
    if sofa is None:
        raise _AttemptFailed("sofa slot failed")

    facing = _facing_of(sofa)
    # coffee table 0.4-0.6 m in front of the sofa's front face
    table_asset = placer.pick_asset("coffee_table")
    table = None
    if table_asset:
        table_dims = placer.dims_of(table_asset)
        half_x, half_y = rotated_half_extents(table_dims, sofa.rot_z_deg)
        for _ in range(SLOT_ATTEMPTS):
            distance = rng.uniform(0.4, 0.6)
            if facing[1] != 0.0:  # sofa faces +/-Y: its front face is the top/bottom of its box
                front = sofa.aabb[3] if facing[1] > 0 else sofa.aabb[1]
                center_y = front + facing[1] * (distance + half_y)
                center_x = sofa.pos[0] + rng.uniform(-0.15, 0.15)
            else:
                front = sofa.aabb[2] if facing[0] > 0 else sofa.aabb[0]
                center_x = front + facing[0] * (distance + half_x)
                center_y = sofa.pos[1] + rng.uniform(-0.15, 0.15)
            item = placer.make("coffee_table", "coffee_table", table_asset, (center_x, center_y),
                               sofa.rot_z_deg, checks_mod.FREE_STAND)
            if item is not None and placer.fits(item, checks_mod.WALL_GAP_FREE_MIN_M):
                table = placer.add(item)
                break
    # rug under the sofa front + coffee table
    rug = None
    if table is not None:
        along_axis = "x" if facing[1] != 0.0 else "y"
        room_along = rect.w if along_axis == "x" else rect.h
        room_across = rect.h if along_axis == "x" else rect.w
        rug_along = min(2.6, room_along - 0.6)
        rug_across = min(1.5, room_across - 0.6)
        for _ in range(SLOT_ATTEMPTS):
            if along_axis == "x":
                size = [rug_along, rug_across]
                center = (sofa.pos[0] + rng.uniform(-0.1, 0.1), table.pos[1] - facing[1] * 0.35)
                rot = 0.0 if facing[1] != 0.0 else 90.0
            else:
                size = [rug_across, rug_along]
                center = (table.pos[0] - facing[0] * 0.35, sofa.pos[1] + rng.uniform(-0.1, 0.1))
                rot = 90.0 if facing[1] != 0.0 else 0.0
            item = placer.make("rug", "rug", RUG_ASSET, center, rot, checks_mod.FLOOR_FLAT, size=size)
            if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M, keep_doors=False):
                rug = placer.add(item)
                break
    # bookshelf on another wall
    shelf_asset = placer.pick_asset("bookshelf")
    shelf = None
    if shelf_asset:
        shelf_dims = placer.dims_of(shelf_asset)
        other_walls = [wall for wall in placer.walls.values()
                       if not _same_wall(wall, sofa)]
        other_walls.sort(key=lambda wall: (placer.wall_has_window(wall), -wall.length))
        shelf = _place_on_wall(placer, "bookshelf", "bookshelf", shelf_asset, other_walls,
                               avoid=[sofa.aabb] + ([table.aabb] if table else []))
    # plant in a free corner
    plant_asset = placer.pick_asset("plant_tall") or placer.pick_asset("plant_small")
    plant = None
    if plant_asset:
        plant_dims = placer.dims_of(plant_asset)
        candidates = []
        inset = checks_mod.WALL_GAP_FREE_MIN_M + 0.10
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            candidates.append((rect.x0 + inset + plant_dims[0] / 2.0 if sx > 0 else rect.x1 - inset - plant_dims[0] / 2.0,
                               rect.y0 + inset + plant_dims[1] / 2.0 if sy > 0 else rect.y1 - inset - plant_dims[1] / 2.0))
        plant = _place_free(placer, "plant", "plant", plant_asset, candidates, attempts=4)
    # armchair facing the seating group
    armchair = None
    if want_armchair:
        chair_asset = placer.pick_asset("armchair")
        if chair_asset:
            chair_dims = placer.dims_of(chair_asset)
            candidates = _free_candidates(placer, chair_dims, step=0.3)
            target = table.pos if table is not None else sofa.pos

            def _face_seating(pos, target=target):
                return _rot_towards((target[0] - pos[0], target[1] - pos[1]))

            armchair = _place_free(placer, "armchair", "armchair", chair_asset, candidates,
                                   attempts=40, rot_fn=_face_seating)
    # low basket in a corner
    basket_asset = placer.pick_asset("basket")
    if basket_asset and rng.random() < 0.6:
        basket_dims = placer.dims_of(basket_asset)
        walls = sorted(placer.walls.values(), key=lambda wall: rng.random())
        _place_on_wall(placer, "basket", "basket", basket_asset, walls)
    # ceiling lamp at the room centre
    lamp_asset = placer.pick_asset("ceiling_lamp")
    if lamp_asset:
        center = (rect.center[0] + rng.uniform(-0.3, 0.3), rect.center[1] + rng.uniform(-0.3, 0.3))
        item = placer.make("ceiling_lamp", "ceiling_lamp", lamp_asset, center, 0.0, checks_mod.CEILING)
        if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M, keep_doors=False):
            placer.add(item)
    # art above the sofa
    art_asset = placer.pick_asset("art")
    if art_asset:
        wall = _facing_wall(placer, sofa)
        item = placer.against_wall("art", "art_sofa", art_asset, wall, sofa.pos[0] if wall.side in ("N", "S") else sofa.pos[1],
                                   placement=checks_mod.MOUNTED, z=1.62, gap=ART_GAP_M)
        if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M, keep_doors=False):
            placer.add(item)


def _rot_towards(direction: Tuple[float, float]) -> float:
    dx, dy = direction
    if abs(dx) >= abs(dy):
        return 90.0 if dx > 0 else 270.0
    return 180.0 if dy > 0 else 0.0


def _facing_of(item: Item) -> Tuple[float, float]:
    """Which way an item's front points, from its rotation."""
    return {0: (0.0, -1.0), 90: (1.0, 0.0), 180: (0.0, 1.0), 270: (-1.0, 0.0)}[int(item.rot_z_deg) % 360]


def _same_wall(wall: Wall, item: Item) -> bool:
    """True when the item's back is against ``wall`` (it faces into the room)."""
    return wall.facing == _facing_of(item)


def _facing_wall(placer: Placer, item: Item) -> Wall:
    back = {0.0: "N", 90.0: "W", 180.0: "S", 270.0: "E"}[item.rot_z_deg % 360.0]
    return placer.walls[back]


def furnish_bedroom(placer: Placer, want_extras: bool) -> None:
    rng = placer.rng
    rect = placer.rect
    bed_asset = placer.pick_asset("bed")
    if bed_asset is None:
        raise _AttemptFailed("no bed asset available")
    bed_dims = placer.dims_of(bed_asset)
    door_sides = []
    for door in placer.doors:
        if door.wall.side in ("N", "S") and abs(door.wall.line - rect.y1) < 1e-6:
            door_sides.append("N")
        elif door.wall.side in ("N", "S") and abs(door.wall.line - rect.y0) < 1e-6:
            door_sides.append("S")
        elif door.wall.side in ("W", "E") and abs(door.wall.line - rect.x1) < 1e-6:
            door_sides.append("E")
        elif door.wall.side in ("W", "E") and abs(door.wall.line - rect.x0) < 1e-6:
            door_sides.append("W")
    bed = None
    free_walls = [wall for wall in placer.walls.values() if wall.side not in door_sides]
    free_walls.sort(key=lambda wall: (placer.wall_has_window(wall), -wall.length, rng.random()))
    door_walls = [wall for wall in placer.walls.values() if wall.side in door_sides]
    door_walls.sort(key=lambda wall: (placer.wall_has_window(wall), -wall.length, rng.random()))
    for wall in free_walls + door_walls:
        across, along_extent = bed_dims[1], bed_dims[0]
        intervals = placer.along_intervals(wall, along_extent, across, bed_dims[2], margin=0.20)
        if not intervals:
            continue
        for _ in range(SLOT_ATTEMPTS):
            along = placer.sample_along(intervals)
            item = placer.against_wall("bed", "bed", bed_asset, wall, along)
            if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M) \
                    and placer.keeps_room_open(item):
                bed = placer.add(item)
                break
        if bed is not None:
            break
    if bed is None:
        raise _AttemptFailed("bed slot failed")

    wall = _facing_wall(placer, bed)
    nightstand_asset = placer.pick_asset("nightstand")
    nightstand = None
    if nightstand_asset:
        stand_dims = placer.dims_of(nightstand_asset)
        along = bed.pos[0] if wall.side in ("N", "S") else bed.pos[1]
        bed_half = bed_dims[0] / 2.0
        for sign in (1.0, -1.0):
            offset = along + sign * (bed_half + 0.06 + stand_dims[0] / 2.0)
            if not wall.contains(offset, stand_dims[0] / 2.0 + 0.05):
                continue
            item = placer.against_wall("nightstand", "nightstand", nightstand_asset, wall, offset)
            if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M):
                nightstand = placer.add(item)
                break
    if nightstand is not None:
        lamp = placer.make("table_lamp", "bedside_lamp", LAMP_ASSET, nightstand.pos, 0.0,
                           checks_mod.MOUNTED, z=round(nightstand.dims[2], 4),
                           attached_to="nightstand")
        if lamp is not None:
            placer.add(lamp)
    art_asset = placer.pick_asset("art")
    if art_asset:
        along = bed.pos[0] if wall.side in ("N", "S") else bed.pos[1]
        item = placer.against_wall("art", "art_bed", art_asset, wall, along,
                                   placement=checks_mod.MOUNTED, z=1.62, gap=ART_GAP_M)
        if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M, keep_doors=False):
            placer.add(item)
    if want_extras:
        plant_asset = placer.pick_asset("plant_small")
        if plant_asset and rng.random() < 0.7:
            plant_dims = placer.dims_of(plant_asset)
            inset = checks_mod.WALL_GAP_FREE_MIN_M + 0.10
            corners = []
            for sx in (1, -1):
                for sy in (1, -1):
                    corners.append((rect.x0 + inset + plant_dims[0] / 2.0 if sx > 0
                                    else rect.x1 - inset - plant_dims[0] / 2.0,
                                    rect.y0 + inset + plant_dims[1] / 2.0 if sy > 0
                                    else rect.y1 - inset - plant_dims[1] / 2.0))
            _place_free(placer, "plant", "plant_bedroom", plant_asset, corners, attempts=4)
        elif rng.random() < 0.5:
            shelf_asset = placer.pick_asset("bookshelf")
            if shelf_asset:
                shelf_dims = placer.dims_of(shelf_asset)
                others = [wall for wall in placer.walls.values() if not _same_wall(wall, bed)]
                _place_on_wall(placer, "bookshelf", "bookshelf_bedroom", shelf_asset, others,
                               avoid=[bed.aabb])
    lamp_asset = placer.pick_asset("ceiling_lamp")
    if lamp_asset:
        center = (rect.center[0] + rng.uniform(-0.25, 0.25), rect.center[1] + rng.uniform(-0.25, 0.25))
        item = placer.make("ceiling_lamp", "ceiling_lamp_bed", lamp_asset, center, 0.0, checks_mod.CEILING)
        if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M, keep_doors=False):
            placer.add(item)


def furnish_plain(placer: Placer) -> None:
    """Kitchen / bathroom / study: a plant when it fits, no cabinetry in v1."""
    rng = placer.rng
    rect = placer.rect
    if placer.room.room_type == "study":
        furnish_study(placer)
        return
    chance = 0.75 if placer.room.room_type == "kitchen" else 0.35
    if rng.random() > chance:
        return
    asset = placer.pick_asset("plant_small")
    if asset is None:
        return
    dims = placer.dims_of(asset)
    inset = checks_mod.WALL_GAP_FREE_MIN_M + 0.10
    corners = []
    for sx in (1, -1):
        for sy in (1, -1):
            corners.append((rect.x0 + inset + dims[0] / 2.0 if sx > 0 else rect.x1 - inset - dims[0] / 2.0,
                            rect.y0 + inset + dims[1] / 2.0 if sy > 0 else rect.y1 - inset - dims[1] / 2.0))
    _place_free(placer, "plant", "plant", asset, corners, attempts=4)


def furnish_study(placer: Placer) -> None:
    rng = placer.rng
    rect = placer.rect
    shelf_asset = placer.pick_asset("bookshelf")
    if shelf_asset:
        shelf_dims = placer.dims_of(shelf_asset)
        walls = sorted(placer.walls.values(), key=lambda wall: (placer.wall_has_window(wall), -wall.length))
        _place_on_wall(placer, "bookshelf", "bookshelf_study", shelf_asset, walls)
    plant_asset = placer.pick_asset("plant_small")
    if plant_asset and rng.random() < 0.6:
        dims = placer.dims_of(plant_asset)
        inset = checks_mod.WALL_GAP_FREE_MIN_M + 0.10
        corners = []
        for sx in (1, -1):
            for sy in (1, -1):
                corners.append((rect.x0 + inset + dims[0] / 2.0 if sx > 0 else rect.x1 - inset - dims[0] / 2.0,
                                rect.y0 + inset + dims[1] / 2.0 if sy > 0 else rect.y1 - inset - dims[1] / 2.0))
        _place_free(placer, "plant", "plant_study", plant_asset, corners, attempts=4)
    lamp_asset = placer.pick_asset("ceiling_lamp")
    if lamp_asset:
        center = (rect.center[0] + rng.uniform(-0.25, 0.25), rect.center[1] + rng.uniform(-0.25, 0.25))
        item = placer.make("ceiling_lamp", "ceiling_lamp_study", lamp_asset, center, 0.0, checks_mod.CEILING)
        if item is not None:
            placer.add(item)


def furnish_rooms(rng: random.Random, plans: Sequence[RoomPlan], doors: Sequence[DoorPlan],
                  windows: Dict[str, List[dict]], index: Dict[str, dict]) -> Tuple[List[Item], List[dict]]:
    placed: List[Item] = []
    report: dict = {"slots": [], "rejections": {}}
    for plan in plans:
        placer = Placer(plan, doors, windows.get(plan.room_id, []), index, rng, placed, report)
        if plan.room_type == "living_room":
            furnish_living(placer, want_armchair=plan.rect.area >= 19.0)
        elif plan.room_type == "bedroom":
            furnish_bedroom(placer, want_extras=plan.rect.area >= 13.0)
        else:
            furnish_plain(placer)
        for reason, count in placer.rejections.items():
            key = f"{plan.room_id}:{reason}"
            report["rejections"][key] = report["rejections"].get(key, 0) + count
    return placed, report["slots"]


# --------------------------------------------------------------------------- #
# step 5: analytic shell model + camera placement
# --------------------------------------------------------------------------- #
class _Box:
    """Axis-aligned box, optionally with holes that let a ray pass through."""

    __slots__ = ("bmin", "bmax", "holes")

    def __init__(self, bmin, bmax, holes=()):
        self.bmin = bmin
        self.bmax = bmax
        self.holes = holes

    def hit(self, origin, direction, t_min: float = 1e-4):
        t0, t1 = t_min, float("inf")
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if origin[axis] < self.bmin[axis] or origin[axis] > self.bmax[axis]:
                    return None
                continue
            inv = 1.0 / direction[axis]
            ta = (self.bmin[axis] - origin[axis]) * inv
            tb = (self.bmax[axis] - origin[axis]) * inv
            if ta > tb:
                ta, tb = tb, ta
            t0 = max(t0, ta)
            t1 = min(t1, tb)
            if t0 > t1:
                return None
        if t1 <= t_min:
            return None
        for hole in self.holes:
            if hole(origin, direction, t0, t1):
                return None
        return max(t0, t_min)


def _sample_row(lat: float, height: int) -> int:
    """The pixel row the validator samples for ``lat``.

    ``tools/validate_blender_agent_p0.py`` maps a latitude to a row with
    ``(pi/2 - lat) / pi * height - 0.5``, rounds it and clamps it to
    ``[2, height - 3]``; the model has to do the same or its rays drift from the
    sampled ones by up to half a row (0.09 deg at 1024 rows), which is exactly
    the margin that decides whether a wall ray grazes over a doorway lintel.
    """
    return max(2, min(height - 3, int(round((math.pi / 2.0 - lat) / math.pi * height - 0.5))))


def _door_hole(door_span_xy, door_height):
    """Predicate: does this ray travel through the door tunnel?"""
    x0, x1, y0, y1 = door_span_xy

    def predicate(origin, direction, t0, t1):
        for t in (t0, t1):
            point = (origin[0] + direction[0] * t, origin[1] + direction[1] * t, origin[2] + direction[2] * t)
            if not (x0 <= point[0] <= x1 and y0 <= point[1] <= y1):
                return False
            if point[2] > door_height:
                return False
        return True

    return predicate


class ShellModel:
    """Analytic stand-in for the built shell (walls with door tunnels, floors,
    ceilings) plus the furniture boxes, used to pre-compute the validator's
    door depth spot check."""

    def __init__(self, rooms: Dict[str, Rect], doors: Sequence[DoorPlan], items: Sequence[Item],
                 wall_height: float = WALL_HEIGHT, wall_thickness: float = WALL_THICKNESS):
        self.room_list = list(rooms.values())
        self.boxes: List[_Box] = []
        self.wall_height = wall_height
        room_specs = [spec_mod.RoomSpec(id=room_id, type="room", polygon=rect.polygon())
                      for room_id, rect in rooms.items()]
        self.segments = [(segment.p0, segment.p1, segment.direction, segment.owners)
                         for segment in spec_mod.build_wall_segments(room_specs)]
        self.door_segments: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        half = wall_thickness / 2.0
        for p0, p1, direction, owners in self.segments:
            (ax, ay), (bx, by) = p0, p1
            bmin = [min(ax, bx) - half, min(ay, by) - half, 0.0]
            bmax = [max(ax, bx) + half, max(ay, by) + half, wall_height]
            holes = []
            for door in doors:
                if not (all(room_id in owners for room_id in (door.a, door.b))):
                    continue
                low, high = door.span
                if door.wall.side in ("N", "S"):
                    span = (low, high, min(ay, by) - half - 1e-3, max(ay, by) + half + 1e-3)
                else:
                    span = (min(ax, bx) - half - 1e-3, max(ax, bx) + half + 1e-3, low, high)
                holes.append(_door_hole(span, DOOR_H))
                self.door_segments[door.key()] = (p0, p1)
            self.boxes.append(_Box(tuple(bmin), tuple(bmax), tuple(holes)))
        for item in items:
            aabb = item.aabb
            z0, z1 = item.z_range
            self.boxes.append(_Box((aabb[0], aabb[1], z0), (aabb[2], aabb[3], z1)))

    def cast(self, origin, direction, t_min: float = 1e-3) -> float:
        best = float("inf")
        for box in self.boxes:
            hit = box.hit(origin, direction, t_min)
            if hit is not None and hit < best:
                best = hit
        for rect in self.room_list:
            for z in (0.0, self.wall_height):
                if abs(direction[2]) < 1e-12:
                    continue
                t = (z - origin[2]) / direction[2]
                if t <= t_min or t >= best:
                    continue
                x = origin[0] + direction[0] * t
                y = origin[1] + direction[1] * t
                if rect.x0 - 1e-9 <= x <= rect.x1 + 1e-9 and rect.y0 - 1e-9 <= y <= rect.y1 + 1e-9:
                    best = t
        return best

    def _row_depth(self, origin, lon: float, row: int, height: int) -> float:
        """Depth along the ERP ray of one pixel row (measured at the row centre)."""
        lat = math.pi / 2.0 - (row + 0.5) / height * math.pi
        ray = (math.cos(lat) * math.cos(lon), -math.cos(lat) * math.sin(lon), math.sin(lat))
        return self.cast(tuple(origin), ray)

    def _window_median(self, origin, lon: float, row: int, height: int, half: int = 2) -> float:
        """Median row depth over the validator's pixel window.

        The validator takes the median depth of a 5x5 pixel neighbourhood around
        the sampled column and row, so the forecast has to be the median of those
        *rows* too -- a single centre ray can slide under a doorway lintel that
        most of the window still sees, and then predicts a wall depth the render
        never reports.  The column spread is left out: at 2 px it is a ~7 cm
        lateral offset, while the rows decide the lintel.
        """
        rows = [max(0, min(height - 1, row + offset)) for offset in range(-half, half + 1)]
        depths = sorted(self._row_depth(origin, lon, value, height) for value in rows)
        return depths[len(depths) // 2]

    def ray_for_pixel(self, door: DoorPlan, camera_pos: Sequence[float],
                      height: int = PANO_H) -> Optional[dict]:
        """Depths of the validator's door/wall rows for the column it samples.

        ``tools/validate_blender_agent_p0.py`` computes the column from
        ``atan2(dy, dx)`` while the ERP convention (``spec.py``) maps a direction
        to ``atan2(-dy, dx)``: the sampled column is the mirror image about the
        camera's +X axis.  The model reproduces that, so the forecast matches the
        render even where the column is mirrored, and it samples the same pixel
        rows (:func:`_sample_row`) with the same window median.
        """
        segment = self.door_segments.get(door.key())
        if segment is None:
            return None
        direction_xy = segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]
        norm = math.hypot(*direction_xy)
        direction_xy = (direction_xy[0] / norm, direction_xy[1] / norm)
        normal = (-direction_xy[1], direction_xy[0])
        to_door = (door.center[0] - camera_pos[0], door.center[1] - camera_pos[1])
        length = math.hypot(*to_door)
        if length < 1e-6:
            return None
        unit = (to_door[0] / length, to_door[1] / length)
        denominator = unit[0] * normal[0] + unit[1] * normal[1]
        if abs(denominator) < 1e-6:
            return None
        horizontal = ((door.center[0] - camera_pos[0]) * normal[0]
                      + (door.center[1] - camera_pos[1]) * normal[1]) / denominator
        if horizontal <= 1e-6:
            return None
        column = int(round((math.atan2(to_door[1], to_door[0]) + math.pi) / (2.0 * math.pi) * PANO_W - 0.5)) % PANO_W
        lon = (column + 0.5) / PANO_W * 2.0 * math.pi - math.pi
        row_door = _sample_row(math.atan2(DOOR_H / 2.0 - camera_pos[2], horizontal), height)
        row_wall = _sample_row(
            math.atan2(((DOOR_H + 0.1 + self.wall_height - 0.05) / 2.0) - camera_pos[2], horizontal), height)
        depths = {}
        for name, row in (("door", row_door), ("wall", row_wall)):
            depths[name] = self._window_median(camera_pos, lon, row, height)
        if not (0 < depths["door"] < float("inf") and 0 < depths["wall"] < float("inf")):
            return None
        return {"column": column, "horizontal_m": horizontal, "door_depth_m": depths["door"],
                "wall_depth_m": depths["wall"], "ratio": depths["door"] / depths["wall"]}

def door_forecast(model: ShellModel, door: DoorPlan, camera_pos: Sequence[float]) -> Optional[dict]:
    """The validator's door spot check, pre-computed on the analytic shell."""
    return model.ray_for_pixel(door, tuple(camera_pos))


def _blocks_walking(item: Item) -> bool:
    """Mirror of ``checks.ItemBox.blocks_walking``: does the item block walkers?"""
    if item.placement == checks_mod.FLOOR_FLAT:
        return False
    return item.z_range[0] < checks_mod.WALK_BAND_M[1] and item.z_range[1] > checks_mod.WALK_BAND_M[0]


def _blocks_camera(item: Item) -> bool:
    if item.placement in (checks_mod.FLOOR_FLAT, checks_mod.CEILING, checks_mod.MOUNTED):
        return False
    return item.z_range[0] < checks_mod.WALK_BAND_M[1] and item.z_range[1] > checks_mod.WALK_BAND_M[0]


def _unit(vector: Sequence[float]) -> Tuple[float, float]:
    length = math.hypot(vector[0], vector[1])
    if length < 1e-9:
        return (0.0, 0.0)
    return (vector[0] / length, vector[1] / length)


def _bearing_angle(first: Sequence[float], second: Sequence[float]) -> float:
    """Angle in degrees between two view bearings (how differently two cameras look)."""
    dot = max(-1.0, min(1.0, first[0] * second[0] + first[1] * second[1]))
    return math.degrees(math.acos(dot))


def _cell_span(low: float, high: float, origin: float, step: float, count: int) -> Tuple[int, int]:
    first = max(0, int(math.ceil((low - origin) / step - 1e-9)))
    last = min(count - 1, int(math.floor((high - origin) / step + 1e-9)))
    return first, last


def _walkable_grid(plan: RoomPlan, items: Sequence[Item], step: float = checks_mod.GRID_M) -> dict:
    """One room's free-cell lattice plus the cells its doorways can reach.

    The rule is the one :func:`checks.circulation_report` applies to the whole
    apartment: a cell is *free* when it is inside the room polygon **inset by the
    wall half thickness** (the walkable side of the wall centre line) and not
    inside a blocking footprint dilated by ``checks.FURNITURE_DILATION_M`` -- rugs,
    wall art and ceiling lamps never block.  ``reachable`` is a BFS over the free
    cells from this room's doorways, so a camera standing on a reachable cell is
    walkable from the front door, which is what the circulation check requires.
    """
    poly = checks_mod.inset_polygon(plan.rect.polygon(), WALL_THICKNESS / 2.0)
    xs = [float(point[0]) for point in poly]
    ys = [float(point[1]) for point in poly]
    x0, y0 = min(xs) - step, min(ys) - step
    nx = int(math.ceil((max(xs) + step - x0) / step)) + 1
    ny = int(math.ceil((max(ys) + step - y0) / step)) + 1
    free = bytearray(nx * ny)
    for ix in range(nx):
        cx = x0 + ix * step
        for iy in range(ny):
            cy = y0 + iy * step
            if checks_mod.point_in_polygon((cx, cy), poly):
                free[ix * ny + iy] = 1
    blocked = bytearray(nx * ny)
    for item in items:
        if not _blocks_walking(item):
            continue
        aabb = item.aabb
        ix_lo, ix_hi = _cell_span(aabb[0] - checks_mod.FURNITURE_DILATION_M,
                                  aabb[2] + checks_mod.FURNITURE_DILATION_M, x0, step, nx)
        iy_lo, iy_hi = _cell_span(aabb[1] - checks_mod.FURNITURE_DILATION_M,
                                  aabb[3] + checks_mod.FURNITURE_DILATION_M, y0, step, ny)
        for ix in range(ix_lo, ix_hi + 1):
            for iy in range(iy_lo, iy_hi + 1):
                blocked[ix * ny + iy] = 1
    return {"x0": x0, "y0": y0, "step": step, "nx": nx, "ny": ny, "free": free, "blocked": blocked,
            "reachable": bytearray(nx * ny)}


def _grid_cell(grid: dict, x: float, y: float) -> Optional[Tuple[int, int]]:
    ix = int(round((x - grid["x0"]) / grid["step"]))
    iy = int(round((y - grid["y0"]) / grid["step"]))
    if 0 <= ix < grid["nx"] and 0 <= iy < grid["ny"]:
        return ix, iy
    return None


def _grid_free(grid: dict, ix: int, iy: int) -> bool:
    index = ix * grid["ny"] + iy
    return bool(grid["free"][index]) and not grid["blocked"][index]


def _walkable(grid: dict, x: float, y: float, reachable: bool = False) -> bool:
    cell = _grid_cell(grid, x, y)
    if cell is None or not _grid_free(grid, cell[0], cell[1]):
        return False
    return bool(grid["reachable"][cell[0] * grid["ny"] + cell[1]]) if reachable else True


def _mark_reachable(grid: dict, starts: Sequence[Sequence[float]]) -> None:
    """BFS over the free cells from the points nearest ``starts`` (the doorways)."""
    reachable = grid["reachable"]
    ny, nx = grid["ny"], grid["nx"]
    queue: List[Tuple[int, int]] = []
    for start in starts:
        cell = _grid_cell(grid, start[0], start[1])
        if cell is None:
            continue
        found = None
        for radius in range(0, 8):  # same spiral as checks.room_connectivity
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    jx, jy = cell[0] + dx, cell[1] + dy
                    if 0 <= jx < nx and 0 <= jy < ny and _grid_free(grid, jx, jy):
                        found = (jx, jy)
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            continue
        index = found[0] * ny + found[1]
        if not reachable[index]:
            reachable[index] = 1
            queue.append(found)
    while queue:
        ix, iy = queue.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < nx and 0 <= jy < ny):
                continue
            index = jx * ny + jy
            if reachable[index] or not _grid_free(grid, jx, jy):
                continue
            reachable[index] = 1
            queue.append((jx, jy))


def _camera_stands(plan: RoomPlan, items: Sequence[Item], grid: dict,
                   rows: Sequence[float]) -> List[Tuple[float, float]]:
    """Candidate stands: the walkable grid sampled every ``CAMERA_GRID_M`` in X
    and on the given door-line rows in Y.

    A stand has to sit on a free cell the room's doorway BFS reaches (so the
    circulation check passes), keep ``CAMERA_INSET_M`` off the wall centre lines
    and ``CAMERA_MIN_GAP_TO_FURNITURE_M`` off every blocking footprint.
    """
    rect = plan.rect
    blockers = [item for item in items if _blocks_camera(item)]
    stands: List[Tuple[float, float]] = []
    x = rect.x0 + CAMERA_INSET_M
    while x <= rect.x1 - CAMERA_INSET_M + 1e-9:
        for row in rows:
            point = (round(x, 3), round(row, 3))
            if not (rect.y0 + CAMERA_INSET_M - 1e-9 <= point[1] <= rect.y1 - CAMERA_INSET_M + 1e-9):
                continue
            if _walkable(grid, point[0], point[1], reachable=True):
                box = (point[0], point[1], point[0], point[1])
                if checks_mod.wall_gap(box, rect.polygon()) >= CAMERA_INSET_M - 1e-9 and \
                        all(checks_mod.rect_distance(box, item.aabb) >= CAMERA_MIN_GAP_TO_FURNITURE_M - 1e-9
                            for item in blockers):
                    stands.append(point)
        x += CAMERA_GRID_M
    return stands


def _coverage_targets(grid: dict, plan: RoomPlan) -> List[Tuple[float, float]]:
    """The room's walkable floor sampled every ``CAMERA_VIS_GRID_M``.

    These are the points a stand's *coverage* is measured on: the share of them
    whose sight line at ``CAMERA_TARGET_Z_M`` reaches them unobstructed.
    """
    rect = plan.rect
    margin = WALL_THICKNESS / 2.0 + 0.05
    targets: List[Tuple[float, float]] = []
    x = rect.x0 + margin
    while x <= rect.x1 - margin + 1e-9:
        y = rect.y0 + margin
        while y <= rect.y1 - margin + 1e-9:
            if _walkable(grid, x, y):
                targets.append((round(x, 3), round(y, 3)))
            y += CAMERA_VIS_GRID_M
        x += CAMERA_VIS_GRID_M
    return targets


def _ray_to(origin, point) -> Tuple[Optional[Tuple[float, float, float]], float]:
    dx, dy, dz = point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance < 1e-9:
        return None, 0.0
    return (dx / distance, dy / distance, dz / distance), distance


def _visible(model: ShellModel, origin, point) -> bool:
    """True when nothing stands between the camera and ``point``."""
    direction, distance = _ray_to(origin, point)
    if direction is None:
        return False
    return model.cast(origin, direction) >= distance - CAMERA_VIS_TOLERANCE_M


def _ray_box_entry(origin, direction, aabb, z_range, t_max: float) -> Optional[float]:
    """Distance at which the ray enters the furniture box (slab test), or None."""
    low = (aabb[0], aabb[1], z_range[0])
    high = (aabb[2], aabb[3], z_range[1])
    t0, t1 = 0.0, t_max
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < low[axis] or origin[axis] > high[axis]:
                return None
            continue
        inv = 1.0 / direction[axis]
        near, far = (low[axis] - origin[axis]) * inv, (high[axis] - origin[axis]) * inv
        if near > far:
            near, far = far, near
        t0, t1 = max(t0, near), min(t1, far)
        if t0 > t1:
            return None
    return t0 if t0 > 1e-6 else None


def _room_camera_context(plan: RoomPlan, items: Sequence[Item], doors: Sequence[DoorPlan]) -> dict:
    """Everything a stand needs: the walkable grid, the coverage targets and the
    blocking furniture (with the probe point a visibility test aims at)."""
    grid = _walkable_grid(plan, items)
    room_doors = [door for door in doors if plan.room_id in (door.a, door.b)]
    _mark_reachable(grid, [door.center for door in room_doors] or [plan.rect.center])
    blockers = []
    for item in items:
        if not _blocks_camera(item):
            continue
        aabb = item.aabb
        z_range = item.z_range
        blockers.append({"aabb": aabb, "z_range": z_range,
                         "probe": (0.5 * (aabb[0] + aabb[2]), 0.5 * (aabb[1] + aabb[3]),
                                   _clamp(0.5 * (z_range[0] + z_range[1]), 0.30, 1.30))})
    return {"grid": grid,
            "targets": _coverage_targets(grid, plan),
            "blockers": blockers,
            "door_rects": door_rects_for(plan.rect, room_doors)}


def _camera_view(model: ShellModel, plan: RoomPlan, context: dict, pos: Sequence[float]) -> dict:
    """What a camera standing at ``pos`` sees, and the terms its score is built from."""
    origin = (pos[0], pos[1], CAMERA_Z_M)
    seen, sum_x, sum_y = 0, 0.0, 0.0
    for target in context["targets"]:
        if _visible(model, origin, (target[0], target[1], CAMERA_TARGET_Z_M)):
            seen += 1
            sum_x += target[0]
            sum_y += target[1]
    blockers = context["blockers"]
    items_seen = 0
    for block in blockers:
        direction, distance = _ray_to(origin, block["probe"])
        if direction is None:
            continue
        entry = _ray_box_entry(origin, direction, block["aabb"], block["z_range"], distance)
        if entry is not None and model.cast(origin, direction) >= entry - CAMERA_VIS_TOLERANCE_M:
            items_seen += 1
    box = (pos[0], pos[1], pos[0], pos[1])
    rect = plan.rect
    center = rect.center if not seen else (sum_x / seen, sum_y / seen)
    clearance = min((checks_mod.rect_distance(box, block["aabb"]) for block in blockers),
                    default=float("inf"))
    view = {
        "coverage": seen / max(1, len(context["targets"])),
        "visible_cells": seen,
        "cells": len(context["targets"]),
        "visible_items": items_seen,
        "items": len(blockers),
        "furniture_fraction": (items_seen / len(blockers)) if blockers else 1.0,
        "wall_clearance": checks_mod.wall_gap(box, rect.polygon()),
        "furniture_clearance": clearance,
        "door_gap": min((checks_mod.rect_distance(box, rect_) for rect_ in context["door_rects"]),
                        default=CAMERA_DOOR_GAP_M),
        "bearing": _unit((center[0] - pos[0], center[1] - pos[1])),
    }
    view["score"] = _camera_score(view)
    return view


def _camera_score(view: dict) -> float:
    """Rank a stand: seeing the room beats standing in a corner, a doorway or
    right next to a wardrobe."""
    return (CAMERA_COVERAGE_WEIGHT * view["coverage"]
            + CAMERA_FURNITURE_WEIGHT * view["furniture_fraction"]
            + CAMERA_WALL_WEIGHT * min(view["wall_clearance"], CAMERA_WALL_COMFORT_M) / CAMERA_WALL_COMFORT_M
            + CAMERA_CLEARANCE_WEIGHT * min(view["furniture_clearance"], CAMERA_CLEARANCE_COMFORT_M)
            / CAMERA_CLEARANCE_COMFORT_M
            + CAMERA_DOOR_WEIGHT * min(view["door_gap"], CAMERA_DOOR_GAP_M) / CAMERA_DOOR_GAP_M)


def _forecast_entry(plan: RoomPlan, door: DoorPlan, pick: dict) -> dict:
    """One ``door_visual_forecast`` row: the pre-computed door check plus what the
    stand was picked for."""
    view = pick["view"]
    clearance = view["furniture_clearance"]
    return {
        "room": plan.room_id,
        "ratio": round(pick["ratio"], 3),
        "opening0": door.key(),
        "score": round(view["score"], 3),
        "coverage": {
            "visible_area_fraction": round(view["coverage"], 3),
            "visible_cells": view["visible_cells"],
            "cells": view["cells"],
            "visible_items": view["visible_items"],
            "furniture": view["items"],
            "wall_clearance_m": round(view["wall_clearance"], 3),
            "furniture_clearance_m": None if clearance == float("inf") else round(clearance, 3),
            "door_gap_m": round(view["door_gap"], 3),
            "view_bearing_deg": round(math.degrees(math.atan2(view["bearing"][1], view["bearing"][0])), 1),
        },
    }


def _second_camera_entry(entry: dict, pick: dict) -> dict:
    """Add the relationship to the room's first camera (second camera only)."""
    entry["second_camera"] = {"gap_m": round(pick["gap"], 3), "bearing_angle_deg": round(pick["angle"], 1)}
    return entry


def place_cameras(rng: random.Random, plans: Sequence[RoomPlan], doors: Sequence[DoorPlan],
                  items: Sequence[Item], model: ShellModel,
                  want_second: bool) -> Tuple[List[dict], DoorPlan, dict]:
    """Pick 1-2 cameras per room by sampling the walkable area.

    Every room contributes at least one camera.  Stands come from the room's
    *walkable grid* (the free-cell lattice of :func:`_walkable_grid`, furniture
    footprints dilated as in :mod:`blender_agent.checks`) and are scored by what
    they see: the share of the walkable floor in unobstructed view, how much of
    the furniture is visible, the clearance to walls and furniture and the
    distance to the doorways.  The old rule -- the point closest to the room
    centre -- put a camera in the middle of the room or in a corner behind the
    sofa; the view score picks the stand that shows the room.

    Two constraints survive from the previous design, because the render
    validator depends on them:

    * a stand only counts when the pre-computed door check
      (:func:`door_forecast`) still sees the opening at
      ``DOOR_FORECAST_MIN_RATIO`` -- the validator samples the *mirrored*
      column of the door direction (see :meth:`ShellModel.ray_for_pixel`), which
      only reaches the doorway from the band around the door line, hence
      ``CAMERA_BAND_M``; the opening is chosen to maximise the worst forecast
      over all cameras;
    * the second camera in the living room has to stand
      ``CAMERA_SECOND_MIN_GAP_M`` away from the first and look at the room from a
      different bearing (``CAMERA_SECOND_MIN_ANGLE_DEG``), so a two-camera room
      is really seen from two sides.
    """
    items_by_room: Dict[str, List[Item]] = {plan.room_id: [] for plan in plans}
    for item in items:
        items_by_room[item.room].append(item)
    # stands are sampled on the rows around the common door line (half the grid
    # step apart); the forecast filter below then keeps the ones whose mirrored
    # column really reaches the doorway
    door_line = sum(door.center[1] for door in doors) / max(1, len(doors))
    rows = [door_line + index * CAMERA_GRID_M / 2.0
            for index in range(-CAMERA_ROW_STEPS, CAMERA_ROW_STEPS + 1)]
    contexts: Dict[str, dict] = {}
    for plan in plans:
        context = _room_camera_context(plan, items_by_room[plan.room_id], doors)
        context["stands"] = _camera_stands(plan, items_by_room[plan.room_id], context["grid"], rows)
        if not context["stands"]:
            raise _AttemptFailed(f"no walkable camera stand in room {plan.room_id}")
        contexts[plan.room_id] = context

    views: Dict[Tuple[str, float, float], dict] = {}

    def view_of(plan: RoomPlan, pos: Sequence[float]) -> dict:
        key = (plan.room_id, pos[0], pos[1])
        if key not in views:
            views[key] = _camera_view(model, plan, contexts[plan.room_id], pos)
        return views[key]

    def feasible(plan: RoomPlan, door: DoorPlan, pos: Sequence[float]):
        if abs(pos[1] - door.center[1]) > CAMERA_BAND_M + 1e-9:
            return None
        forecast = door_forecast(model, door, (pos[0], pos[1], CAMERA_Z_M))
        ratio = forecast["ratio"] if forecast else 0.0
        if ratio < DOOR_FORECAST_MIN_RATIO:
            return None
        return ratio

    best_door = None
    best_worst = None
    order = list(doors)
    rng.shuffle(order)  # deterministic tie-break when several openings score alike
    for door in order:
        picks: Dict[str, dict] = {}
        worst = float("inf")
        for plan in plans:
            best = None
            for pos in contexts[plan.room_id]["stands"]:
                ratio = feasible(plan, door, pos)
                if ratio is None:
                    continue
                # the door sight line decides *whether* a stand is usable, the
                # view score decides which one is picked
                value = view_of(plan, pos)["score"] + CAMERA_RATIO_WEIGHT * min(ratio, 3.0)
                if best is None or value > best["value"] + 1e-9:
                    best = {"pos": pos, "ratio": ratio, "view": view_of(plan, pos), "value": value}
            if best is None:
                picks = {}
                break
            picks[plan.room_id] = best
            worst = min(worst, best["ratio"])
        if picks and (best_worst is None or worst > best_worst + 1e-9):
            best_worst = worst
            best_door = (door, picks)
    if best_door is None:
        raise _AttemptFailed("no opening is visible from every camera candidate")

    door, picks = best_door
    cameras: List[dict] = []
    camera_forecast: Dict[str, dict] = {}
    index = 0
    for plan in plans:
        pick = picks[plan.room_id]
        name = f"{index:04d}"
        index += 1
        cameras.append({"name": name, "room": plan.room_id,
                        "pos": [round(pick["pos"][0], 3), round(pick["pos"][1], 3), CAMERA_Z_M]})
        camera_forecast[name] = _forecast_entry(plan, door, pick)
    if want_second:
        living = next((plan for plan in plans if plan.room_type == "living_room"), None)
        if living is not None:
            first = picks[living.room_id]

            def second(plan: RoomPlan, min_angle: float) -> Optional[dict]:
                best = None
                for pos in contexts[plan.room_id]["stands"]:
                    gap = math.dist(pos, first["pos"])
                    if gap < CAMERA_SECOND_MIN_GAP_M - 1e-9:
                        continue
                    ratio = feasible(plan, door, pos)
                    if ratio is None:
                        continue
                    view = view_of(plan, pos)
                    angle = _bearing_angle(first["view"]["bearing"], view["bearing"])
                    if angle < min_angle - 1e-9:
                        continue
                    value = (view["score"] + CAMERA_SECOND_GAP_WEIGHT * min(gap, 3.0)
                             + CAMERA_SECOND_ANGLE_WEIGHT * angle / 180.0)
                    if best is None or value > best["value"] + 1e-9:
                        best = {"pos": pos, "ratio": ratio, "view": view, "value": value,
                                "angle": angle, "gap": gap}
                return best

            # a second camera is only worth it if it really looks at the room from
            # another side; where the furniture leaves no stand that far around,
            # the best remaining stand still wins (the report keeps its angle)
            best = second(living, CAMERA_SECOND_MIN_ANGLE_DEG) or second(living, 0.0)
            if best is not None:
                name = f"{index:04d}"
                cameras.append({"name": name, "room": living.room_id,
                                "pos": [round(best["pos"][0], 3), round(best["pos"][1], 3), CAMERA_Z_M]})
                camera_forecast[name] = _second_camera_entry(_forecast_entry(living, door, best), best)
    return cameras, door, camera_forecast


# --------------------------------------------------------------------------- #
# step 6: assembly
# --------------------------------------------------------------------------- #
def _spec_dict(seed: int, plans: Sequence[RoomPlan], doors: Sequence[DoorPlan],
               windows: Dict[str, List[dict]], cameras: Sequence[dict], items: Sequence[Item],
               opening_order: Sequence[DoorPlan]) -> dict:
    rooms = []
    for plan in plans:
        rooms.append({"id": plan.room_id, "type": plan.room_type, "polygon": plan.rect.polygon(),
                      "floor_color": ROOM_FLOOR_COLOR.get(plan.room_type, ROOM_FLOOR_COLOR_DEFAULT)})
    openings = []
    for door in opening_order:
        openings.append({"kind": "door", "between": [door.a, door.b],
                         "center": [round(door.center[0], 4), round(door.center[1], 4)],
                         "width": DOOR_W, "height": DOOR_H})
    windows_out = []
    for plan in plans:
        for entry in windows.get(plan.room_id, []):
            windows_out.append({"room": entry["room"], "wall": entry["wall"],
                                "sill": entry["sill"], "height": entry["height"]})
    return {
        "_comment": [
            f"generated by blender_agent/layout_gen.py --seed {seed}",
            "a row of axis-aligned rooms (repeated vertical guillotine cuts), one door per",
            "shared wall on a common door line near the south edge, windows on exterior wall",
            "spans only, furniture placed by room type with rot_z_deg on the 90 degree grid",
            "(front faces -Y at 0, see assets.py).  See the layout report for the door",
            "visibility forecast that the depth validation measures after the render.",
        ],
        "unit": "meter",
        "wall_height": WALL_HEIGHT,
        "wall_thickness": WALL_THICKNESS,
        "rooms": rooms,
        "openings": openings,
        "windows": windows_out,
        "cameras": cameras,
        "style": dict(STYLE_DEFAULTS),
        "materials": json.loads(json.dumps(MATERIAL_DEFAULTS)),
        "details": dict(DETAIL_DEFAULTS),
        "lighting": dict(LIGHTING_DEFAULTS),
        "furniture": [item.to_json() for item in items],
    }


@dataclass
class Scene:
    spec: dict
    report: dict


def _build_attempt(rng: random.Random, room_count: Optional[int], index: Dict[str, dict],
                   attempt: int, seed: int = 0) -> Tuple[dict, dict]:
    count = room_count or rng.choice([2, 3, 3, 4, 4, 4, 5, 5])
    count = int(_clamp(count, 2, 5))
    types = choose_room_types(rng, count)
    plans = plan_rooms(rng, types)
    doors = plan_doors(rng, plans)
    windows = plan_windows(rng, plans)
    items, slot_report = furnish_rooms(rng, plans, doors, windows, index)
    rooms_by_id = {plan.room_id: plan.rect for plan in plans}
    model = ShellModel(rooms_by_id, doors, items)
    want_second = count <= 3
    cameras, opening0, forecast = place_cameras(rng, plans, doors, items, model, want_second)
    order = [opening0] + [door for door in doors if door is not opening0]
    spec = _spec_dict(seed, plans, doors, windows, cameras, items, order)
    spec["_comment"].append(f"rooms={count} attempt={attempt} furniture={len(items)}")
    report = {
        "rooms": [{"id": plan.room_id, "type": plan.room_type,
                   "rect": [plan.rect.x0, plan.rect.y0, plan.rect.x1, plan.rect.y1],
                   "area_m2": round(plan.rect.area, 2),
                   "polygon": plan.rect.polygon()} for plan in plans],
        "room_graph": {"nodes": [plan.room_id for plan in plans],
                       "edges": [[door.a, door.b] for door in doors]},
        "doors": [{"between": [door.a, door.b], "center": [round(door.center[0], 3), round(door.center[1], 3)],
                   "width": DOOR_W, "height": DOOR_H, "wall_side": door.wall.side} for door in doors],
        "windows": [{"room": entry["room"], "wall": entry["wall"], "sill": entry["sill"],
                     "height": entry["height"]} for plan in plans for entry in windows.get(plan.room_id, [])],
        "cameras": cameras,
        "furniture": [{"name": item.name, "asset": item.asset, "room": item.room,
                       "pos": [round(item.pos[0], 3), round(item.pos[1], 3)],
                       "rot_z_deg": item.rot_z_deg, "z": item.z, "mount": item.mount,
                       "size_m": [round(v, 3) for v in item.dims], "placement": item.placement,
                       "slot": item.slot, "attached_to": item.attached_to} for item in items],
        "furniture_slots": slot_report,
        "opening_0": opening0.key(),
        "door_visual_forecast": forecast,
    }
    return spec, report


def generate(seed: int, room_count: Optional[int] = None,
             assets_index: Optional[Dict[str, dict]] = None,
             attempts: int = MAX_LAYOUT_ATTEMPTS, verbose: bool = True) -> Scene:
    """Generate one layout; returns the spec dict plus the layout report."""
    index = assets_index if assets_index is not None else checks_mod.load_assets_index()
    started = time.time()
    last_error = "unknown"
    for attempt in range(attempts):
        rng = random.Random(seed * 1000003 + attempt)
        try:
            spec_dict, report = _build_attempt(rng, room_count, index, attempt, seed)
            spec = spec_mod.parse_spec(spec_dict)
        except (_AttemptFailed, spec_mod.SpecError) as error:
            last_error = f"{type(error).__name__}: {error}"
            continue
        check_report = checks_mod.check_spec(spec, index)
        report["checks"] = check_report
        if not check_report["pass"]:
            last_error = "; ".join(check_report["violations"][:4]) or "checks failed"
            continue
        report["stats"] = {
            "seed": seed,
            "attempt": attempt,
            "attempts_used": attempt + 1,
            "generation_seconds": round(time.time() - started, 3),
            "rooms": len(spec.rooms),
            "doors": len(spec.openings),
            "windows": len(spec.windows),
            "cameras": len(spec.cameras),
            "furniture": len(spec.furniture),
        }
        if verbose:
            print(f"[layout_gen] seed {seed}: {report['stats']} (last rejected attempt: {last_error})"
                  if attempt else f"[layout_gen] seed {seed}: {report['stats']}")
        return Scene(spec=spec_dict, report=report)
    raise LayoutError(f"seed {seed}: no layout in {attempts} attempts (last failure: {last_error})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate a furnished apartment scene spec from a seed")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True, help="scene_spec JSON to write")
    parser.add_argument("--rooms", type=int, default=None, help="2..5, default: drawn from the seed")
    parser.add_argument("--assets-index", default=DEFAULT_ASSETS_INDEX)
    parser.add_argument("--report", default=None, help="layout report (default: next to --out)")
    parser.add_argument("--attempts", type=int, default=MAX_LAYOUT_ATTEMPTS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.rooms is not None and not 2 <= args.rooms <= 5:
        parser.error("--rooms must be between 2 and 5")

    index = checks_mod.load_assets_index(args.assets_index)
    try:
        scene = generate(args.seed, args.rooms, index, attempts=args.attempts, verbose=not args.quiet)
    except LayoutError as error:
        print(f"[layout_gen] FAILED: {error}", file=sys.stderr)
        return 1

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(scene.spec, handle, indent=2)
    report_path = os.path.abspath(args.report) if args.report else \
        os.path.join(os.path.dirname(out_path), DEFAULT_REPORT_NAME)
    scene.report["spec"] = out_path
    scene.report["seed"] = args.seed
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(scene.report, handle, indent=2)
    if not args.quiet:
        checks_path = os.path.join(REPO_ROOT, "blender_agent", "checks.py")
        print(f"[layout_gen] spec   : {out_path}")
        print(f"[layout_gen] report : {report_path}")
        print(f"[layout_gen] checks : PASS ({checks_path} {out_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
