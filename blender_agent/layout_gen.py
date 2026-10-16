"""Autonomous apartment layout generator: seed in, scene spec out.

    python blender_agent/layout_gen.py --seed 0 --out spec.json \
        [--rooms 2..7] [--shape row|block|auto|<family>] \
        [--assets-index third_party/assets/polyhaven/index.json] \
        [--report layout_report.json]

The generator turns a random seed into a furnished, multi-room ``scene_spec``
that the existing pipeline can build, render and validate:

1. *shell*: axis-aligned rectangular rooms in a **topology family** (see
   :data:`ROOM_FAMILIES`): a 2x2 block, an L, a central living room with rooms
   around it, one- or two-sided corridors and a circulation loop.  ``--shape``
   picks the arrangement (``row`` is the default railroad flat, ``block`` the
   same chain wrapped into two rows, ``auto`` draws a family from the seed, and
   a family name forces one).  Room areas are drawn from the realistic bands
   (:data:`ROOM_AREA_M2`) with a minimum side per type, so a small service cell
   becomes a bathroom and never a 3 m2 bedroom.
2. *doors*: a semantic spanning tree over the shared walls -- a corridor serves
   the rooms, the kitchen opens onto the dining room, a bathroom is never
   crossed and a bedroom is never the public way through the flat
   (:func:`door_semantics` checks the result).  Loop families add one extra
   door to close their ring.
3. *windows*: 1-2 per habitable room on its own exterior wall spans
   (1.2-2.0 m wide, sill 0.9 m, height 1.4 m), kept clear of wall corners, and
   allocated by the function of the room (:func:`window_wish`): living rooms,
   bedrooms, studies and dining rooms get windows, kitchens nearly always one,
   bathrooms and laundries rarely, corridors and storage none.
4. *furniture*: Poly Haven assets picked by category from the asset index (the
   API dimensions are millimetres and become the collision boxes) plus the
   functional ``builtin:`` furniture of :mod:`blender_agent.fixtures` -- a
   kitchen counter run (cabinets, stove, sink), a fridge, a bathroom set
   (toilet, basin, shower), desk/chair/bookshelf, dining table with chairs,
   wardrobe, shoe cabinet, storage shelving, washing machine.  Every fixture
   claims the clear space its *use* needs (a fridge door swing, a cook's
   standing spot, a pulled-out desk chair) and the layout refuses to put
   anything there.  A failed required slot restarts the layout.
5. *cameras*: 1-2 per room, sampled on the room's **walkable grid** (the free
   cells of :func:`checks.circulation_report`, furniture footprints dilated)
   and picked by what they see: the share of the walkable floor and of the
   furniture in unobstructed view, the clearance to walls and furniture, the
   distance to the doorways and whether the doorway itself is on a clear sight
   line.  Stands stay >= 0.6 m off the furniture, and a two-camera room gets
   two stands that look at it from different bearings.
6. *verification*: :mod:`blender_agent.checks` must pass, and the door
   visibility of every camera is pre-computed with an analytic shell model
   (:class:`ShellModel`) through the same world->camera->pixel projection the
   validator uses, so the rendered report and the forecast agree.

No door line, no camera row
---------------------------
This generator used to *force* every door and every camera onto one common Y
line (and to keep the middle of every room free of furniture), because the
validator's door spot check located its sample column with ``atan2(dy, dx)``.
Under the ``R0`` convention of ``spec.py`` a direction becomes the column
``atan2(-dy, dx)``, so that formula sampled the *mirror image* about the
camera's +X axis: a target to the north-east (1, 1) was sampled to the
south-east (0.707, -0.707), a target to the north to the south.  On a row with
one shared door line the mirror maps the plan onto itself and the check
measured the real sight line; in a general 2-D plan it sampled a wall and
reported a ratio near 1.0 for a perfectly good room.

Both sides are now derived from the camera matrix instead: the validator
projects a world point with the inverse camera rotation
(``tools/validate_blender_agent_p0.py``: :func:`world_to_erp`) and this module
does the same arithmetic in pure Python (:func:`erp_pixel_of_direction`), which
is what :meth:`ShellModel.ray_for_pixel` uses to pre-compute the door sight
line.  The door geometry itself is accepted per opening with local rays from
both sides of the wall, so the plan does not have to be a row for the
validation to be meaningful.


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
from blender_agent import fixtures as fixtures_mod  # noqa: E402
from blender_agent import spec as spec_mod  # noqa: E402

DEFAULT_ASSETS_INDEX = checks_mod.DEFAULT_ASSETS_INDEX
DEFAULT_REPORT_NAME = "layout_report.json"
PANO_W = spec_mod.PANO_WIDTH
PANO_H = spec_mod.PANO_HEIGHT

WALL_HEIGHT = 2.7
WALL_THICKNESS = 0.12
DOOR_W = 0.9
DOOR_H = 2.1
DOOR_JAMB_M = 0.20
WINDOW_SILL = 0.9
WINDOW_H = 1.4
WINDOW_W_RANGE = (1.2, 2.0)
WINDOW_MARGIN_M = 0.35

HABITABLE = ("living_room", "bedroom", "study", "kitchen", "dining")
# Room-type contract of the generator.  The bands are the realism envelope the
# family planners draw within (see ``ROOM_FAMILIES``); a layout whose rectangle
# lands outside its type's band or below the minimum side fails the attempt and
# is redrawn, which is what keeps a 2x2 block from squeezing a 3 m2 "bedroom".
ROOM_AREA_M2 = {
    "living_room": (14.0, 30.0),
    "bedroom": (10.0, 20.0),
    "kitchen": (5.0, 14.0),
    "bathroom": (3.2, 8.5),
    "study": (6.0, 14.0),
    "dining": (5.0, 16.0),
    "corridor": (3.0, 16.0),
    "entry": (2.0, 8.0),
    "storage": (1.5, 6.0),
    "balcony": (2.0, 8.0),
    "laundry": (1.8, 6.0),
}
ROOM_MIN_DIM_M = {"living_room": 3.2, "bedroom": 2.7, "kitchen": 2.0, "bathroom": 1.7,
                  "study": 2.4, "dining": 2.2, "corridor": 1.1, "entry": 1.2,
                  "storage": 1.0, "balcony": 1.2, "laundry": 1.4}
ROOM_FLOOR_COLOR = {
    "living_room": [0.55, 0.38, 0.22],
    "bedroom": [0.60, 0.50, 0.40],
    "kitchen": [0.62, 0.61, 0.58],
    "bathroom": [0.72, 0.73, 0.74],
    "study": [0.50, 0.36, 0.24],
    "dining": [0.56, 0.42, 0.28],
    "corridor": [0.52, 0.47, 0.40],
    "entry": [0.66, 0.63, 0.58],
    "storage": [0.60, 0.56, 0.50],
    "balcony": [0.45, 0.47, 0.45],
    "laundry": [0.68, 0.69, 0.70],
}
ROOM_FLOOR_COLOR_DEFAULT = [0.58, 0.50, 0.42]
#: room type -> id stem (a second room of a type becomes ``stem_2``)
ROOM_ID_BASE = {
    "living_room": "living", "bedroom": "bedroom", "kitchen": "kitchen",
    "bathroom": "bathroom", "study": "study", "dining": "dining",
    "corridor": "corridor", "entry": "entry", "storage": "storage",
    "balcony": "balcony", "laundry": "laundry",
}
#: rooms nobody should have to walk through to reach another room
PRIVATE_ROOM_TYPES = ("bedroom", "bathroom")

GRID_STEP_M = 0.05
SNAP_M = 0.05
MIN_SHARED_WALL_M = 1.4
ROOM_DEPTH_MIN_M = 3.65
ROOM_DEPTH_MAX_M = 4.80
ROOM_AREA_HEADROOM_M = 0.30
ROOM_OPEN_FRACTION = 0.85
MAX_LAYOUT_ATTEMPTS = 80
SLOT_ATTEMPTS = 8
CAMERA_INSET_M = 0.40
CAMERA_GRID_M = 0.25
CAMERA_Z_M = 1.5
CAMERA_MIN_GAP_TO_FURNITURE_M = 0.60
CAMERA_DOOR_GAP_M = 0.60
CAMERA_DOOR_SIGHT_WEIGHT = 0.35
CAMERA_DOOR_SIGHT_TOLERANCE_M = 0.10
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
# fixture geometry lives in blender_agent/fixtures.py as ``builtin:`` parts
FIXTURE_GAP_M = 0.005          # gap between the modules of a counter run
USAGE_GAP_M = 0.02             # tolerance of a keep-clear zone in front of a fixture
USAGE_NEIGHBOUR_M = 0.10       # an item this close to a fixture is beside it, not in front
UPPER_CABINET_Z_M = 1.62       # wall cupboards start above checks.WALK_BAND_M[1]

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


def erp_pixel_of_direction(direction: Sequence[float], rotation: Sequence[Sequence[float]] = None,
                           width: int = PANO_W, height: int = PANO_H) -> Tuple[int, int]:
    """Panorama pixel of a world direction: ``(column, row)``.

    The pure-Python twin of ``tools/validate_blender_agent_p0.py``'s
    :func:`world_to_erp` and the only place here that turns a direction into a
    column.  The camera matrix is *inverted* (a rotation is orthonormal, so its
    inverse is its transpose) and ``lon``/``lat`` are read off the local vector
    that ``spec.py``'s ERP convention defines: a pixel shoots
    ``R @ (cos(lat) sin(lon), -sin(lat), cos(lat) cos(lon))``.  The old code
    instead wrote ``atan2(dy, dx)`` by hand, which is the mirror image of that
    column under ``R0`` -- the reason every door and camera used to sit on one
    common Y line.
    """
    rotation = spec_mod.R0 if rotation is None else rotation
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
    local = (rotation[0][0] * dx + rotation[1][0] * dy + rotation[2][0] * dz,
             rotation[0][1] * dx + rotation[1][1] * dy + rotation[2][1] * dz,
             rotation[0][2] * dx + rotation[1][2] * dy + rotation[2][2] * dz)
    lon = math.atan2(local[0], local[2])
    lat = math.atan2(-local[1], math.hypot(local[0], local[2]))
    column = int(round((lon + math.pi) / (2.0 * math.pi) * width - 0.5)) % width
    row = int(round((math.pi / 2.0 - lat) / math.pi * height - 0.5))
    return column, max(0, min(height - 1, row))


def erp_ray_direction(lon: float, lat: float,
                      rotation: Sequence[Sequence[float]] = None) -> Tuple[float, float, float]:
    """World ray of an ERP pixel centre: the forward direction of the same map."""
    rotation = spec_mod.R0 if rotation is None else rotation
    local = (math.cos(lat) * math.sin(lon), -math.sin(lat), math.cos(lat) * math.cos(lon))
    return tuple(sum(rotation[i][k] * local[k] for k in range(3)) for i in range(3))


def erp_lon_lat(column: int, row: int, width: int = PANO_W, height: int = PANO_H):
    """Pixel centre -> ``(lon, lat)``, exactly as ``spec.py``'s ERP map does."""
    return ((column + 0.5) / width * 2.0 * math.pi - math.pi,
            math.pi / 2.0 - (row + 0.5) / height * math.pi)


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
    """The functional programme of a home: living room, bedroom, then the rest.

    Every home gets a living room and a bedroom, three rooms and more get a
    kitchen, four and more a bathroom; the remaining slots are drawn from the
    optional rooms (a second bedroom, study, dining room, storage, entry hall,
    laundry, balcony).  Order is drawn from the seed -- the *arrangement* of the
    programme is the topology family's job (:data:`ROOM_FAMILIES`).
    """
    types = ["living_room"]
    if count >= 2:
        types.append("bedroom")
    if count >= 3:
        types.append("kitchen")
    if count >= 4:
        types.append("bathroom")
    pool = ["bedroom", "bedroom", "study", "study", "dining", "storage", "entry", "laundry", "balcony"]
    used: Dict[str, int] = {"bedroom": 1 if count >= 2 else 0, "kitchen": 1 if count >= 3 else 0,
                            "bathroom": 1 if count >= 4 else 0}
    guard = 0
    while len(types) < count and guard < 400:
        guard += 1
        pick = rng.choice(pool)
        if pick == "bedroom" and used.get("bedroom", 0) >= 2:
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

    This is the default *arrangement*, not a constraint of the pipeline: it is
    kept because a 2-5 room flat in one row is a common real plan, and
    :func:`plan_rooms_block` wraps the same chain of rooms around two rows for
    an L / block plan (``--shape block``).  Nothing else in the generator (doors,
    furniture, camera stands, the door forecast) assumes a row any more -- the
    row used to be forced by the validator's mirrored door column, which is
    fixed (see the module docstring).
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
    # private rooms (bedroom / bathroom) take the ends of the chain wherever the
    # programme allows it: a middle room of a chain is a passage, so a bedroom
    # there would be the public way through the flat (see :func:`door_semantics`)
    rng.shuffle(plans)
    private = [plan for plan in plans if plan.room_type in PRIVATE_ROOM_TYPES]
    public = [plan for plan in plans if plan.room_type not in PRIVATE_ROOM_TYPES]
    split = len(private) // 2
    plans = private[:split] + public + private[split:]
    x = 0.0
    for plan in plans:
        x1 = _round(x + plan.width)
        plan.rect = Rect(x, 0.0, x1, _round(plan.depth, 0.01))
        x = x1
    if x < 6.0:
        raise _AttemptFailed("row too short")
    ids: Dict[str, int] = {}
    for plan in plans:
        base = ROOM_ID_BASE[plan.room_type]
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


def plan_rooms_block(rng: random.Random, types: Sequence[str]) -> List[RoomPlan]:
    """The same chain of rooms as :func:`plan_rooms`, wrapped into two rows.

    The chain -- and with it the room graph, the door graph and every furniture
    slot -- is untouched: the first half of the chain runs along the south edge
    as usual, the second half turns the corner at the end of it and runs back
    *above* the first row, so consecutive rooms still share a full wall
    (:func:`plan_doors` finds every one of them) and the plan is two-dimensional
    -- an L for three rooms, a 2x2 block for four, a longer block for five.
    Rooms keep their drawn area; each row gets one depth (the deepest room of
    the row) and the widths are recomputed from the areas, which is what makes
    the two rows share a straight wall.

    This is the "no door line" shape's reason to exist: with the validator
    measuring doors per opening, a plan no longer has to put every room on one X
    axis for the door check to be meaningful.
    """
    plans = plan_rooms(rng, types)
    bottom_count = max(1, len(plans) // 2)
    bottom, top = plans[:bottom_count], plans[bottom_count:]
    if not top:  # a single room cannot be wrapped; keep the row
        return plans

    def row_depth(row: Sequence[RoomPlan]) -> float:
        low, high = ROOM_DEPTH_MIN_M, ROOM_DEPTH_MAX_M
        depth = _clamp(max(plan.depth for plan in row), low, high)
        for plan in row:
            width = plan.target_area / depth
            if width < ROOM_MIN_DIM_M[plan.room_type] - 1e-9:
                raise _AttemptFailed(f"{plan.room_type} cannot keep its area in a {depth:.2f} m row")
        return depth

    placements: List[Tuple[RoomPlan, Rect]] = []
    x = 0.0
    depth = row_depth(bottom)
    for plan in bottom:
        x1 = _round(x + plan.target_area / depth)
        placements.append((plan, Rect(x, 0.0, x1, _round(depth, 0.01))))
        x = x1
    if x < 6.0:
        raise _AttemptFailed("the ground row is too short")
    corner = placements[-1][1]
    x, y0 = corner.x1, corner.y1
    depth = row_depth(top)
    for plan in top:
        width = _round(plan.target_area / depth)
        placements.append((plan, Rect(x - width, y0, x, _round(y0 + depth, 0.01))))
        x -= width

    for plan, rect in placements:
        plan.rect = rect
        low, high = ROOM_AREA_M2[plan.room_type]
        if plan.rect.w < ROOM_MIN_DIM_M[plan.room_type] - 1e-9 or \
                plan.rect.h < ROOM_MIN_DIM_M[plan.room_type] - 1e-9:
            raise _AttemptFailed(f"room {plan.room_type} too small "
                                 f"({plan.rect.w:.2f}x{plan.rect.h:.2f} m)")
        if not (low - 1e-9 <= plan.rect.area <= high + 1e-9):
            raise _AttemptFailed(f"room {plan.room_type} area {plan.rect.area:.1f} m2 outside "
                                 f"[{low}, {high}] after the block snap")
    return plans


# --------------------------------------------------------------------------- #
# step 1b: topology families
# --------------------------------------------------------------------------- #
# A room is always a rectangle; a family builds the *arrangement* -- a 2x2
# block, an L, a middle living room, one- or two-sided corridors, a circulation
# loop -- as a small grid of cells (columns x rows) with one room per span of
# cells.  The pattern is fixed, the cell sizes are drawn per attempt, so the
# same family gives a different flat for every seed; the room bands of
# ``ROOM_AREA_M2`` are the envelope a draw has to land in or the attempt is
# retried.

#: role -> room types it can host (the first entry is the usual reading)
ROLE_POOLS: Dict[str, Tuple[str, ...]] = {
    "public": ("living_room",),
    "cook": ("kitchen",),
    "sleep": ("bedroom",),
    "sleep2": ("bedroom", "study"),
    "wet": ("bathroom",),
    "wash": ("laundry", "storage"),
    "work": ("study", "storage"),
    "eat": ("dining", "study"),
    "flow": ("corridor", "entry"),
    "entry": ("entry", "storage"),
    "outdoor": ("balcony", "laundry"),
}


@dataclass(frozen=True)
class _Slot:
    """One room of a family template, in grid coordinates.

    ``col``/``row`` are 0-based cell indices, ``col_span``/``row_span`` how many
    cells the room covers (a living room spanning two rows is one rectangle).
    """

    role: str
    col: int
    row: int
    col_span: int = 1
    row_span: int = 1


@dataclass(frozen=True)
class _Family:
    """A topology template: cell size ranges plus one slot per room."""

    name: str
    cols: Tuple[Tuple[float, float], ...]
    rows: Tuple[Tuple[float, float], ...]
    slots: Tuple[_Slot, ...]
    extra_doors: int = 0           # doors beyond the spanning tree (ring layouts)

    @property
    def count(self) -> int:
        return len(self.slots)


ROOM_FAMILIES: Dict[str, _Family] = {
    # 4 rooms in a 2x2 block: living with kitchen beside it, bedroom and
    # bathroom behind (the bathroom's only allowed wall is the bedroom's --
    # no door runs from a kitchen into a bathroom, see door_semantics)
    "compact_2x2": _Family(
        name="compact_2x2",
        cols=((3.7, 4.3), (2.1, 2.6)),
        rows=((3.5, 4.0), (2.9, 3.4)),
        slots=(_Slot("public", 0, 0), _Slot("cook", 1, 0),
                _Slot("sleep", 0, 1), _Slot("wet", 1, 1)),
    ),
    # an L: a dining room and a narrow kitchen column beside it, the living room
    # below the dining room, the bathroom off the living room and the bedroom
    # closing the far end
    "l_shape": _Family(
        name="l_shape",
        cols=((4.0, 5.0), (2.2, 2.8)),
        rows=((3.2, 3.8), (3.0, 3.6), (3.0, 3.6)),
        slots=(_Slot("eat", 0, 0), _Slot("cook", 1, 0),
                _Slot("public", 0, 1), _Slot("wet", 1, 1),
                _Slot("sleep", 0, 2)),
    ),
    # central living room spanning the middle band, rooms north and south of it
    # -- every room opens off the living room (the "radiating" plan)
    "central_living": _Family(
        name="central_living",
        cols=((3.4, 4.4), (2.0, 2.6)),
        rows=((2.9, 3.6), (3.2, 4.0), (2.9, 3.6)),
        slots=(_Slot("cook", 0, 0), _Slot("eat", 1, 0),
                _Slot("public", 0, 1, col_span=2),
                _Slot("sleep", 0, 2), _Slot("wet", 1, 2)),
        extra_doors=1,           # kitchen <-> dining as well as via the living room
    ),
    # single-loaded corridor: a hall strip along the south edge, rooms over it,
    # the living room closing the end
    "corridor_single": _Family(
        name="corridor_single",
        cols=((1.9, 2.6), (3.0, 3.8), (2.5, 3.2), (4.0, 5.0)),
        rows=((1.1, 1.6), (2.9, 3.6)),
        slots=(_Slot("flow", 0, 0, col_span=3),
                _Slot("wet", 0, 1), _Slot("sleep", 1, 1), _Slot("cook", 2, 1),
                _Slot("public", 3, 0, row_span=2)),
    ),
    # double-loaded corridor: rooms on both sides of a middle hall
    "corridor_double": _Family(
        name="corridor_double",
        cols=((3.4, 4.2), (2.2, 2.8)),
        rows=((3.4, 4.2), (1.1, 1.6), (2.9, 3.6)),
        slots=(_Slot("public", 0, 0), _Slot("cook", 1, 0),
                _Slot("flow", 0, 1, col_span=2),
                _Slot("sleep", 0, 2), _Slot("wet", 1, 2)),
    ),
    # circulation loop: living -> hall -> kitchen -> dining -> living is a ring
    # (four doors) around a 2x2 block, with the bedroom off the hall and its
    # bathroom as the en-suite leaf
    "loop": _Family(
        name="loop",
        cols=((3.4, 4.2), (2.8, 3.4), (3.0, 3.6)),
        rows=((2.4, 2.9), (3.4, 4.0)),
        slots=(_Slot("eat", 0, 0), _Slot("cook", 1, 0),
                _Slot("public", 0, 1), _Slot("flow", 1, 1),
                _Slot("sleep", 2, 1), _Slot("wet", 2, 0)),
        extra_doors=1,           # the ring's fourth edge, the tree drops one
    ),
}


def _check_room_bands(plan: RoomPlan, where: str) -> None:
    """The shared contract: a room rectangle must land in its type's band."""
    low, high = ROOM_AREA_M2[plan.room_type]
    if plan.rect.w < ROOM_MIN_DIM_M[plan.room_type] - 1e-9 or \
            plan.rect.h < ROOM_MIN_DIM_M[plan.room_type] - 1e-9:
        raise _AttemptFailed(f"room {plan.room_type} too small "
                             f"({plan.rect.w:.2f}x{plan.rect.h:.2f} m{where})")
    if not (low - 1e-9 <= plan.rect.area <= high + 1e-9):
        raise _AttemptFailed(f"room {plan.room_type} area {plan.rect.area:.1f} m2 outside "
                             f"[{low}, {high}]{where}")


def _rect_fits_type(rect: Rect, room_type: str) -> bool:
    """Would a room of this rectangle satisfy its type's band and minimum side?"""
    low, high = ROOM_AREA_M2[room_type]
    if rect.w < ROOM_MIN_DIM_M[room_type] - 1e-9 or rect.h < ROOM_MIN_DIM_M[room_type] - 1e-9:
        return False
    return low - 1e-9 <= rect.area <= high + 1e-9


def build_family(rng: random.Random, family: _Family) -> List[RoomPlan]:
    """One draw of a family: cell sizes, an optional mirror, the room rectangles.

    Column widths and row depths are drawn once for the family's grid and every
    room takes the span of cells its slot covers, so neighbouring cells share
    exactly one boundary line (the shared wall).  A random mirror in X and Y
    varies the orientation without touching the adjacency pattern.  The room
    *type* of a slot is drawn from its role's pool, restricted to the types
    whose area band and minimum side the drawn rectangle satisfies -- a small
    service cell becomes a bathroom, never a 3 m2 bedroom.
    """
    widths = [rng.uniform(low, high) for low, high in family.cols]
    depths = [rng.uniform(low, high) for low, high in family.rows]
    mirror_x, mirror_y = rng.random() < 0.5, rng.random() < 0.5
    n_cols, n_rows = len(widths), len(depths)
    if mirror_x:
        widths.reverse()
    if mirror_y:
        depths.reverse()
    xs = [0.0]
    for width in widths:
        xs.append(_round(xs[-1] + width))
    ys = [0.0]
    for depth in depths:
        ys.append(_round(ys[-1] + depth))

    plans: List[RoomPlan] = []
    used_types: List[str] = []
    for slot in family.slots:
        col = (n_cols - slot.col - slot.col_span) if mirror_x else slot.col
        row = (n_rows - slot.row - slot.row_span) if mirror_y else slot.row
        rect = Rect(xs[col], ys[row], xs[col + slot.col_span], ys[row + slot.row_span])
        fitting = [room_type for room_type in ROLE_POOLS[slot.role] if _rect_fits_type(rect, room_type)]
        if not fitting:
            raise _AttemptFailed(f"no {slot.role} room fits {rect.w:.2f}x{rect.h:.2f} m "
                                 f"({rect.area:.1f} m2) in family '{family.name}'")
        # a type the flat does not have yet beats a second one when both fit
        fresh = [room_type for room_type in fitting if room_type not in used_types]
        room_type = rng.choice(fresh or fitting)
        used_types.append(room_type)
        plan = RoomPlan(room_id="", room_type=room_type, rect=rect, target_area=rect.area,
                        width=rect.w, depth=rect.h)
        _check_room_bands(plan, f" in family '{family.name}'")
        plans.append(plan)
    ids: Dict[str, int] = {}
    for plan in plans:
        base = ROOM_ID_BASE[plan.room_type]
        ids[base] = ids.get(base, 0) + 1
        plan.room_id = base if ids[base] == 1 else f"{base}_{ids[base]}"
    return plans


def _shapes_for_count(count: int) -> Tuple[str, ...]:
    """Every arrangement that can host a flat of ``count`` rooms."""
    names = ["row"]
    if count >= 3:
        names.append("block")
    names += sorted(name for name, family in ROOM_FAMILIES.items() if family.count == count)
    return tuple(names)


def _pick_auto_shape(rng: random.Random, room_count: Optional[int]) -> Tuple[int, str]:
    """Draw a room count and one of its families from the seed ("--shape auto")."""
    draw = [2, 3, 3, 4, 4, 4, 5, 5, 6]
    count = int(_clamp(room_count if room_count is not None else rng.choice(draw), 2, 7))
    return count, rng.choice(_shapes_for_count(count))


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


def plan_doors(rng: random.Random, plans: Sequence[RoomPlan],
               chain: Optional[Sequence[RoomPlan]] = None) -> List[DoorPlan]:
    """One door per shared wall between consecutive rooms of the chain.

    The rooms are a chain (a row, or the two-row L of :func:`plan_rooms_block`),
    so the shared-wall graph is a path and a spanning tree is simply "every
    shared wall gets a door".  Each door is sampled anywhere along its shared
    wall span (with a jamb's margin to the corners), *not* on a common door
    line: the position used to be pinned to one Y so that the validator's
    mirrored sample column could see through it, and the validator now projects
    the door itself (see the module docstring).
    """
    ordered = list(chain) if chain is not None else sorted(plans, key=lambda plan: plan.rect.x0)
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
        raise _AttemptFailed("the rooms are not a chain of shared walls")
    doors: List[DoorPlan] = []
    for index, _next, wall, lo, hi in pairs:
        margin = DOOR_W / 2.0 + 0.20
        if hi - lo < DOOR_W + 2 * 0.20:
            raise _AttemptFailed("shared wall too short for a door")
        center_along = _round(rng.uniform(lo + margin, hi - margin), 0.01)
        # the door centre is a point *on the wall line*: for a north/south wall the
        # line is a Y and the position along it is an X (the old code, which only
        # ever saw east/west walls of the row, wrote the two swapped)
        center = (center_along, wall.line) if wall.side in ("N", "S") else (wall.line, center_along)
        direction = (1.0, 0.0) if wall.side in ("N", "S") else (0.0, 1.0)
        doors.append(DoorPlan(a=ordered[index].room_id, b=ordered[index + 1].room_id,
                              center=center,
                              span=(center_along - DOOR_W / 2.0, center_along + DOOR_W / 2.0),
                              wall=wall, direction=direction))
    return doors


# --------------------------------------------------------------------------- #
# step 2b: doors on a general adjacency graph (the topology families)
# --------------------------------------------------------------------------- #
#: doorway preference between room classes: the score Kruskal picks the spanning
#: tree with.  Keys are sorted ``(class_a, class_b)`` pairs; negative pairs are
#: never connected (a bedroom is not entered from a kitchen, two bedrooms do not
#: open into each other).
DOOR_CLASS_SCORE: Dict[Tuple[str, str], float] = {
    ("hall", "sleep"): 4.0, ("hall", "wet"): 4.0, ("hall", "living"): 3.4,
    ("dining", "hall"): 3.2, ("hall", "kitchen"): 3.0, ("hall", "other"): 3.4,
    ("hall", "hall"): 0.5,
    ("kitchen", "living"): 3.2, ("dining", "living"): 3.0, ("living", "sleep"): 3.0,
    ("living", "wet"): 2.2, ("living", "other"): 2.2, ("living", "living"): 0.4,
    ("dining", "kitchen"): 4.2, ("kitchen", "other"): 1.0,
    ("kitchen", "wet"): -1.0, ("kitchen", "sleep"): -1.0, ("kitchen", "kitchen"): 0.4,
    ("dining", "sleep"): 2.0, ("dining", "wet"): 1.2, ("dining", "other"): 1.6,
    ("dining", "dining"): 0.4,
    ("sleep", "wet"): 1.6, ("other", "wet"): 1.4, ("wet", "wet"): -1.0,
    ("other", "sleep"): 0.4, ("sleep", "sleep"): -1.0, ("other", "other"): 0.6,
}


def _door_class(room_type: str) -> str:
    """The vocabulary :data:`DOOR_CLASS_SCORE` is written in."""
    if room_type in ("corridor", "entry"):
        return "hall"
    if room_type == "living_room":
        return "living"
    if room_type == "bedroom":
        return "sleep"
    if room_type == "bathroom":
        return "wet"
    if room_type in ("kitchen", "dining"):
        return room_type
    return "other"


def _door_score(a_type: str, b_type: str) -> float:
    return DOOR_CLASS_SCORE.get(tuple(sorted((_door_class(a_type), _door_class(b_type)))), 0.5)


def _plans_by_id(plans: Sequence[RoomPlan]) -> Dict[str, RoomPlan]:
    return {plan.room_id: plan for plan in plans}


def _wall_segments_for(plans: Sequence[RoomPlan]):
    room_specs = [spec_mod.RoomSpec(id=plan.room_id, type=plan.room_type, polygon=plan.rect.polygon())
                  for plan in plans]
    return spec_mod.build_wall_segments(room_specs)


def _segment_offset(segments, a_id: str, b_id: str, point):
    """``(segment, offset of ``point`` from its start)`` on a wall owning both rooms."""
    for segment in segments:
        if a_id in segment.owners and b_id in segment.owners and segment.contains_point(point):
            base = spec_mod.line_parameter(segment.p0, segment.direction)
            return segment, spec_mod.line_parameter(point, segment.direction) - base
    return None, None


def plan_doors_graph(rng: random.Random, plans: Sequence[RoomPlan],
                     extra_doors: int = 0) -> List[DoorPlan]:
    """Doors of an arbitrary rectangle plan: a semantic spanning tree + a loop.

    Every pair of rooms that shares a wall long enough is a candidate; Kruskal
    over the doorway preference (:data:`DOOR_CLASS_SCORE`) builds the spanning
    tree, so a corridor serves the rooms, the kitchen opens onto the dining
    room, and bedrooms / bathrooms are only ever *entered* -- their degree is
    capped at one, which is what keeps them off every public path
    (:func:`door_semantics` re-checks it).  ``extra_doors`` adds the
    best-scoring remaining connection between two non-private rooms, which is
    how the loop families get their second way round.
    """
    by_id = _plans_by_id(plans)
    candidates: List[Tuple[float, float, str, str]] = []
    for index, first in enumerate(plans):
        for second in plans[index + 1:]:
            shared = shared_wall_span(first.rect, second.rect)
            if shared is None:
                continue
            _wall, (lo, hi) = shared
            if hi - lo < DOOR_W + 2 * DOOR_JAMB_M:
                continue
            score = _door_score(first.room_type, second.room_type)
            if score < 0.0:
                continue
            jitter = rng.random() * 0.5
            candidates.append((score, jitter, first.room_id, second.room_id))
    candidates.sort(key=lambda entry: (-entry[0], -entry[1], entry[2], entry[3]))
    if len(plans) == 1:
        return []

    degrees = {plan.room_id: 0 for plan in plans}
    parent = {plan.room_id: plan.room_id for plan in plans}

    def find(room_id: str) -> str:
        while parent[room_id] != room_id:
            parent[room_id] = parent[parent[room_id]]
            room_id = parent[room_id]
        return room_id

    def cap(room_id: str) -> int:
        """How many doors a room may have: private rooms are entered, not crossed.

        A bedroom gets a second door only when it leads to its own bathroom (an
        en-suite): that bathroom is a leaf itself, so the bedroom still carries
        nobody from one public room to another (:func:`door_semantics` checks
        the result).
        """
        return 1 if by_id[room_id].room_type in PRIVATE_ROOM_TYPES else 4

    def allowed(a: str, b: str) -> bool:
        if degrees[a] >= cap(a) or degrees[b] >= cap(b):
            en_suite = ((by_id[a].room_type, by_id[b].room_type, degrees[a]) == ("bedroom", "bathroom", 1)
                        or (by_id[b].room_type, by_id[a].room_type, degrees[b]) == ("bedroom", "bathroom", 1))
            if not en_suite:
                return False
        return True

    edges: List[Tuple[str, str]] = []
    for _score, _jitter, a, b in candidates:
        if len(edges) >= len(plans) - 1:
            break
        if not allowed(a, b):
            continue
        if find(a) == find(b):
            continue
        parent[find(a)] = find(b)
        degrees[a] += 1
        degrees[b] += 1
        edges.append((a, b))
    if len(edges) != len(plans) - 1:
        raise _AttemptFailed(f"no spanning door tree over {len(plans)} rooms "
                             f"({len(edges)} of {len(plans) - 1} edges)")
    # extra connections (a circulation loop) close a cycle between public rooms
    used_pairs = {tuple(sorted(edge)) for edge in edges}
    for _score, _jitter, a, b in candidates:
        if len(edges) >= len(plans) - 1 + extra_doors:
            break
        if tuple(sorted((a, b))) in used_pairs:
            continue
        if by_id[a].room_type in PRIVATE_ROOM_TYPES or by_id[b].room_type in PRIVATE_ROOM_TYPES:
            continue
        if not allowed(a, b):
            continue
        used_pairs.add(tuple(sorted((a, b))))
        degrees[a] += 1
        degrees[b] += 1
        edges.append((a, b))

    # place every door on its shared wall, keeping jambs and other doors clear
    segments = _wall_segments_for(plans)
    used_spans: Dict[Tuple[Tuple[float, float], Tuple[float, float]], List[Tuple[float, float]]] = {}
    doors: List[DoorPlan] = []
    for a, b in edges:
        first, second = by_id[a], by_id[b]
        shared = shared_wall_span(first.rect, second.rect)
        if shared is None:
            raise _AttemptFailed(f"edge {a}|{b} no longer shares a wall")
        wall, (lo, hi) = shared
        door = _place_graph_door(rng, first, second, wall, lo, hi, segments, used_spans)
        if door is None:
            raise _AttemptFailed(f"no free place for the door between {a} and {b}")
        doors.append(door)
    return doors


def _place_graph_door(rng: random.Random, first: RoomPlan, second: RoomPlan, wall: Wall,
                      lo: float, hi: float, segments, used_spans) -> Optional[DoorPlan]:
    """Sample one door position on the shared wall that fits the wall segment."""
    low, high = lo + DOOR_W / 2.0 + DOOR_JAMB_M, hi - DOOR_W / 2.0 - DOOR_JAMB_M
    if low > high:
        return None
    for _attempt in range(24):
        along = _round(rng.uniform(low, high), 0.01)
        center = (along, wall.line) if wall.side in ("N", "S") else (wall.line, along)
        segment, offset = _segment_offset(segments, first.room_id, second.room_id, center)
        if segment is None or offset is None:
            continue
        if offset - DOOR_W / 2.0 < DOOR_JAMB_M - 1e-9 or offset + DOOR_W / 2.0 > segment.length - DOOR_JAMB_M + 1e-9:
            continue
        span = (offset - DOOR_W / 2.0, offset + DOOR_W / 2.0)
        key = (segment.p0, segment.p1)
        taken = used_spans.setdefault(key, [])
        if any(span[0] < other[1] - 1e-6 and other[0] < span[1] - 1e-6 for other in taken):
            continue
        taken.append(span)
        return DoorPlan(a=first.room_id, b=second.room_id, center=center,
                        span=(along - DOOR_W / 2.0, along + DOOR_W / 2.0),
                        wall=wall,
                        direction=(1.0, 0.0) if wall.side in ("N", "S") else (0.0, 1.0))
    return None


def door_semantics(plans: Sequence[RoomPlan], doors: Sequence[DoorPlan]) -> dict:
    """The semantic rules the door graph has to satisfy, as a report.

    * every room is reachable (has a door);
    * a bathroom is private: exactly one door, so nobody crosses it;
    * a bedroom is not a public thoroughfare: with all bedrooms and bathrooms
      removed, the remaining (public) rooms are still connected -- a bedroom
      may carry the only door to *its own* en-suite bathroom, but it is never
      the way from one public room to another;
    * the kitchen is next to the dining room when there is one (a shared wall,
      and a door between them whenever the topology offers one).
    """
    by_id = _plans_by_id(plans)
    adjacency = {plan.room_id: set() for plan in plans}
    for door in doors:
        adjacency[door.a].add(door.b)
        adjacency[door.b].add(door.a)
    violations: List[str] = []
    for plan in plans:
        if not adjacency[plan.room_id]:
            violations.append(f"room '{plan.room_id}' ({plan.room_type}) has no door")
    for plan in plans:
        if plan.room_type == "bathroom" and len(adjacency[plan.room_id]) > 1:
            violations.append(f"bathroom '{plan.room_id}' has {len(adjacency[plan.room_id])} doors "
                              f"(a bathroom must stay private)")

    public = [plan.room_id for plan in plans if plan.room_type not in PRIVATE_ROOM_TYPES]
    if public:
        seen = {public[0]}
        frontier = [public[0]]
        while frontier:
            room_id = frontier.pop()
            for peer in adjacency[room_id]:
                if peer in seen or peer not in public:
                    continue
                seen.add(peer)
                frontier.append(peer)
        missing = [room_id for room_id in public if room_id not in seen]
        if missing:
            violations.append("public rooms are only reachable through a bedroom/bathroom: "
                              + ", ".join(missing))

    dining = [plan.room_id for plan in plans if plan.room_type == "dining"]
    kitchen = [plan.room_id for plan in plans if plan.room_type == "kitchen"]
    kitchen_next_to: Optional[str] = None
    kitchen_door_to: Optional[str] = None
    target = dining[0] if dining else (next((plan.room_id for plan in plans
                                             if plan.room_type == "living_room"), None))
    if kitchen and target:
        shared = shared_wall_span(by_id[kitchen[0]].rect, by_id[target].rect)
        if shared is not None:
            kitchen_next_to = by_id[target].room_type
        if by_id[target].room_id in adjacency[kitchen[0]]:
            kitchen_door_to = by_id[target].room_type
        if dining and kitchen_next_to is None:
            violations.append(f"kitchen '{kitchen[0]}' does not share a wall with dining '{target}'")
    for plan in plans:
        if plan.room_type == "bedroom":
            for peer in adjacency[plan.room_id]:
                if by_id[peer].room_type == "kitchen":
                    violations.append(f"bedroom '{plan.room_id}' is entered from the kitchen")

    return {
        "pass": not violations,
        "violations": violations,
        "degrees": {room_id: len(peers) for room_id, peers in sorted(adjacency.items())},
        "bathrooms_are_leaves": all(len(adjacency[plan.room_id]) <= 1
                                    for plan in plans if plan.room_type == "bathroom"),
        "public_rooms_connected": not any("only reachable through" in entry for entry in violations),
        "kitchen_next_to": kitchen_next_to,
        "kitchen_door_to": kitchen_door_to,
    }


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


def window_wish(rng: random.Random, room_type: str, candidates: Sequence) -> int:
    """How many of its exterior spans a room wants a window on.

    Windows follow the real exterior walls (only single-owner spans reach here)
    and the function of the room: living rooms, bedrooms, studies and dining
    rooms are habitable (1-2 windows), a kitchen nearly always has one, a
    bathroom or laundry rarely (0-1), circulation, storage and the entrance hall
    none, and a balcony is a glazed outdoor room (its own window).
    """
    if room_type in HABITABLE:
        return 2 if len(candidates) > 1 and rng.random() < 0.4 else 1
    if room_type == "kitchen":
        return 1 if rng.random() < 0.75 else 0
    if room_type == "bathroom":
        return 1 if rng.random() < 0.25 else 0
    if room_type == "laundry":
        return 1 if rng.random() < 0.30 else 0
    if room_type == "balcony":
        return 1 if rng.random() < 0.9 else 0
    return 0                                   # corridor / entry / storage


def plan_windows(rng: random.Random, plans: Sequence[RoomPlan]) -> Dict[str, List[dict]]:
    """One window per exterior span, per room function (:func:`window_wish`)."""
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
        want = window_wish(rng, plan.room_type, candidates)
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


def _dims_of(asset: str, index: Dict[str, dict],
             size: Optional[Sequence[float]] = None) -> Optional[Tuple[float, float, float]]:
    """Measured footprint of any asset the generator may place.

    ``checks.item_dims`` covers the Poly Haven / MIDI-3D / generated libraries
    and the ``builtin:`` props of ``checks.BUILTIN_DIMS_M``; the functional
    furniture of :mod:`blender_agent.fixtures` is registered there too, and this
    fallback keeps the generator working even when a caller passes its own index.
    """
    dims = checks_mod.item_dims(asset, index, size)
    if dims is None and asset.startswith("builtin:"):
        part = fixtures_mod.FIXTURE_PARTS.get(asset.split(":", 1)[1])
        if part is not None:
            dims = tuple(part["dims"])
    return dims


def usage_rect(item: Item, depth: float,
               behind: bool = False) -> Tuple[float, float, float, float]:
    """The clear space a fixture needs in front of it (door swing / standing spot).

    ``behind`` puts the zone at the item's back instead -- how a chair is pulled
    out from the table it faces.
    """
    aabb = item.aabb
    dir_x, dir_y = _facing_of(item)
    if behind:
        dir_x, dir_y = -dir_x, -dir_y
    if dir_x != 0.0:
        return (aabb[2], aabb[1], aabb[2] + depth, aabb[3]) if dir_x > 0 else \
               (aabb[0] - depth, aabb[1], aabb[0], aabb[3])
    return (aabb[0], aabb[3], aabb[2], aabb[3] + depth) if dir_y > 0 else \
           (aabb[0], aabb[1] - depth, aabb[2], aabb[1])


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
        #: keep-clear zones in front of fixtures (fridge door, pulled-out chair)
        self.usages: List[dict] = []

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
        return _dims_of(asset, self.index)

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

    def fits(self, item: Item, gap: float, keep_doors: bool = True) -> bool:
        aabb = item.aabb
        if not self.inside(aabb, gap):
            self.reject("outside_room")
            return False
        # the wall-gap class of checks.check_spec: wall hugging (<= 0.35 m) or
        # free standing (>= 0.60 m), nothing in between -- a fixture placed by
        # hand (the desk chair, a dining chair) has to land on one side of it
        if item.placement not in (checks_mod.FLOOR_FLAT, checks_mod.CEILING, checks_mod.MOUNTED):
            wall_gap = checks_mod.wall_gap(aabb, self.rect.polygon())
            if wall_gap < checks_mod.WALL_GAP_MIN_M - 1e-9:
                self.reject("wall_gap")
                return False
            if checks_mod.WALL_GAP_HUG_MAX_M + 1e-9 < wall_gap < checks_mod.WALL_GAP_FREE_MIN_M - 1e-9:
                self.reject("wall_gap_band")
                return False
        if keep_doors:
            for rect in self.door_rects():
                if checks_mod.rect_distance(aabb, rect) < checks_mod.DOOR_CLEARANCE_M + 0.05:
                    self.reject("door_zone")
                    return False
        if _blocks_walking(item):
            for usage in self.usages:
                if item.slot and item.slot in usage["allow"]:
                    continue
                if checks_mod.rect_distance(aabb, usage["aabb"]) < USAGE_NEIGHBOUR_M:
                    continue      # this item is beside the fixture, not in its zone
                if checks_mod.rect_distance(aabb, usage["rect"]) < USAGE_GAP_M:
                    self.reject(f"usage:{usage['slot']}")
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

    def add(self, item: Item, usage: float = 0.0, allow: Sequence[str] = (),
            behind: bool = False) -> Optional[Item]:
        """Place the item; ``usage`` registers the keep-clear zone in front of it.

        The zone is claimed *before* the item joins the room: a placement whose
        use would already be blocked by something standing there (a fridge
        behind a shelf) is refused instead of placed with a warning.  ``allow``
        names the slots that may stand in the zone -- the desk's pull-out space
        is exactly where its chair goes; ``behind`` flips the zone to the item's
        back (a chair is pulled out away from the table it faces).
        """
        if usage > 0.0 and _blocks_walking(item):
            rect = usage_rect(item, usage, behind=behind)
            for other in self.placed:
                if not _blocks_walking(other) or (other.slot and other.slot in allow):
                    continue
                if checks_mod.rect_distance(other.aabb, item.aabb) < USAGE_NEIGHBOUR_M:
                    continue      # standing beside the fixture, not in front of it
                if checks_mod.rect_distance(other.aabb, rect) < USAGE_GAP_M:
                    self.reject(f"usage_occupied:{item.slot}")
                    return None
        self.placed.append(item)
        if usage > 0.0 and _blocks_walking(item):
            self.usages.append({"rect": usage_rect(item, usage, behind=behind), "slot": item.slot,
                                "name": item.name, "aabb": item.aabb, "allow": tuple(allow)})
        self.report.setdefault("slots", []).append({
            "slot": item.slot, "name": item.name, "asset": item.asset, "room": item.room,
            "placement": item.placement, "pos": [round(item.pos[0], 3), round(item.pos[1], 3)],
            "rot_z_deg": item.rot_z_deg, "size_m": [round(v, 3) for v in item.dims],
            "usage_m": round(usage, 3) if usage else 0.0,
            "usage_behind": bool(behind),
        })
        return item

    def rollback(self, items: Sequence[Item]) -> None:
        """Undo the placements of a failed group (a counter run that did not fit)."""
        names = {item.name for item in items}
        if not names:
            return
        self.placed[:] = [item for item in self.placed if item.name not in names]
        self.usages[:] = [usage for usage in self.usages if usage["name"] not in names]
        self.report["slots"] = [entry for entry in self.report.get("slots", [])
                                if entry.get("name") not in names]

    def make(self, slot: str, name: str, asset: str, pos, rot, placement: str,
             z: Optional[float] = None, size: Optional[Sequence[float]] = None,
             attached_to: Optional[str] = None) -> Optional[Item]:
        dims = _dims_of(asset, self.index, size)
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
        dims = _dims_of(asset, self.index, size)
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
        """Positions along ``wall`` where the item fits, with the wall margin.

        ``extent_across`` and ``height`` are kept in the signature because every
        furniture slot passes them; they no longer restrict anything.  They used
        to encode the enfilade rule -- a tall item had to stay out of the Y band
        between the doorways so that the validator's mirrored column could see a
        door -- which is gone now that the validator projects the door itself.
        """
        limits = self.along_limits(wall, extent_along / 2.0, margin)
        if limits is None:
            return []
        return [limits]

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
                   attached_to=None, window_penalty: bool = True,
                   usage: float = 0.0, allow: Sequence[str] = ()) -> Optional[Item]:
    """Try each wall (and positions along it) until the item fits."""
    dims = _dims_of(asset, placer.index, size)
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
                placed = placer.add(item, usage=usage, allow=allow)
                if placed is not None:
                    return placed
    return None


def _place_free(placer: Placer, slot: str, name: str, asset: str, candidates,
                placement: str = checks_mod.FREE_STAND, size=None, attempts: int = 26,
                rot_fn=None, usage: float = 0.0, allow: Sequence[str] = ()) -> Optional[Item]:
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
                placed = placer.add(item, usage=usage, allow=allow)
                if placed is not None:
                    return placed
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
    if rect.area >= 11.5 and rng.random() < 0.7:
        walls = [wall for wall in placer.walls.values() if not _same_wall(wall, bed)]
        walls.sort(key=lambda wall: (placer.wall_has_window(wall), -wall.length, rng.random()))
        _place_on_wall(placer, "wardrobe", "wardrobe", _fixture("wardrobe"), walls,
                       usage=fixtures_mod.usage_m("wardrobe"), avoid=[bed.aabb])
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


def _fixture(part: str) -> str:
    """Asset reference of a :mod:`blender_agent.fixtures` part."""
    return f"builtin:{part}"


def _fixture_walls(placer: Placer, prefer_window: bool = False) -> List[Wall]:
    """Walls ordered for fixture placement: without a door first, longest first."""
    door_sides = {door.wall.side for door in placer.doors}
    return sorted(placer.walls.values(),
                  key=lambda wall: (wall.side in door_sides,
                                    0 if placer.wall_has_window(wall) == prefer_window else 1,
                                    -wall.length))


def _place_run(placer: Placer, wall: Wall, parts: Sequence[str], start: float,
               name_prefix: str) -> Optional[List[Item]]:
    """Tile fixture parts along a wall at exact offsets (one straight counter).

    Every part is placed against the wall with ``FIXTURE_GAP_M`` between its
    neighbours, so the worktop of a kitchen run is continuous; the whole run is
    rolled back when any part does not fit, and a part's standing space
    (``fixtures.usage_m``) is claimed before the next part goes down.
    """
    placed: List[Item] = []
    along = float(start)
    for part in parts:
        width = fixtures_mod.dims(part)[0]
        along += width / 2.0
        item = placer.against_wall(part, f"{name_prefix}_{part}", _fixture(part), wall, along)
        if item is None:
            placer.rollback(placed)
            return None
        item.slot = part
        if not placer.fits(item, checks_mod.WALL_GAP_MIN_M) or not placer.keeps_room_open(item):
            placer.rollback(placed)
            return None
        added = placer.add(item, usage=fixtures_mod.usage_m(part))
        if added is None:
            placer.rollback(placed)
            return None
        placed.append(added)
        along += width / 2.0 + FIXTURE_GAP_M
    return placed


def _add_ceiling_lamp(placer: Placer, name: str) -> Optional[Item]:
    """A pendant near the room's centre, retried when a tall piece stands there.

    A windowless room (bathroom, hall, storage) is lit by its doorways alone and
    renders black without a lamp, so this retries a few positions instead of
    giving up when the first spot collides with something like a shower cabin.
    """
    rng = placer.rng
    rect = placer.rect
    lamp_asset = placer.pick_asset("ceiling_lamp")
    if lamp_asset is None:
        return None
    for attempt in range(10):
        offsets = ((0.0, 0.0), (0.30, 0.0), (-0.30, 0.0), (0.0, 0.30), (0.0, -0.30))[attempt % 5]
        center = (rect.center[0] + offsets[0] + rng.uniform(-0.10, 0.10),
                  rect.center[1] + offsets[1] + rng.uniform(-0.10, 0.10))
        item = placer.make("ceiling_lamp", name, lamp_asset, center, 0.0, checks_mod.CEILING)
        if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M, keep_doors=False):
            return placer.add(item)
    return None


def _corner_candidates(placer: Placer, dims, inset: Optional[float] = None):
    rect = placer.rect
    inset = checks_mod.WALL_GAP_FREE_MIN_M + 0.10 if inset is None else inset
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            out.append((rect.x0 + inset + dims[0] / 2.0 if sx > 0 else rect.x1 - inset - dims[0] / 2.0,
                        rect.y0 + inset + dims[1] / 2.0 if sy > 0 else rect.y1 - inset - dims[1] / 2.0))
    return out


def furnish_kitchen(placer: Placer) -> None:
    """A counter run (cabinets + stove + sink) and a fridge, with standing space.

    The run is tiled module by module along the longest door-free wall so the
    worktop reads as one line; the stove and the sink sit in it, the fridge
    closes the line or takes another wall.  Every part claims the clear space
    its use needs (a fridge door swing, a cook's standing spot) -- the generator
    refuses to put anything there, so the fixture stays usable (the render
    validator cannot see intent, which is why the report carries ``usage_m``).
    """
    rng = placer.rng
    widths = {part: fixtures_mod.dims(part)[0] for part in
              ("kitchen_cabinet", "kitchen_stove", "kitchen_sink", "kitchen_fridge")}
    for wall in _fixture_walls(placer, prefer_window=False):
        usable = wall.length - 2.0 * 0.10
        for modules in (4, 3, 2, 1, 0):
            run = ["kitchen_cabinet"] * modules + ["kitchen_stove", "kitchen_sink"]
            run_length = sum(widths[part] for part in run) + FIXTURE_GAP_M * (len(run) - 1)
            fridge_length = widths["kitchen_fridge"] + FIXTURE_GAP_M
            if run_length > usable:
                continue
            with_fridge = run_length + fridge_length <= usable
            total = run_length + fridge_length if with_fridge else run_length
            for _attempt in range(4):
                start = wall.lo + 0.10 + rng.uniform(0.0, max(0.0, usable - total))
                items = _place_run(placer, wall, run, start, "kitchen")
                if items is None:
                    continue
                if with_fridge:
                    _place_run(placer, wall, ["kitchen_fridge"], start + run_length, "kitchen")
                if modules > 0 and not placer.wall_has_window(wall) and rng.random() < 0.5:
                    _place_upper_cabinets(placer, wall, start, run_length)
                _add_ceiling_lamp(placer, "ceiling_lamp_kitchen")
                return
    raise _AttemptFailed("no kitchen run fits on any wall")


def _place_upper_cabinets(placer: Placer, wall: Wall, start: float, length: float) -> None:
    """Wall cupboards above a counter run, high enough to stay out of the walk band."""
    count = 2 if length >= 1.3 else 1
    along = start + length / 2.0
    for index in range(count):
        offset = along + (index - (count - 1) / 2.0) * 0.65
        item = placer.against_wall("kitchen_upper", f"kitchen_upper_{index}", _fixture("kitchen_upper"),
                                   wall, offset, placement=checks_mod.MOUNTED, z=UPPER_CABINET_Z_M)
        if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M, keep_doors=False):
            item.mount = "wall"
            placer.add(item)


def furnish_bathroom(placer: Placer) -> None:
    """Toilet, washbasin and (when the room allows it) a shower cabin."""
    rect = placer.rect
    walls = _fixture_walls(placer, prefer_window=False)
    shower_first = rect.area >= 5.5
    order = ["bath_shower"] if shower_first else []
    order += ["bath_toilet", "bath_basin"]
    if not shower_first and rect.area >= 4.2:
        order += ["bath_shower"]
    for part in order:
        placed = _place_on_wall(placer, part, part, _fixture(part), walls,
                                usage=fixtures_mod.usage_m(part))
        if placed is None and part in ("bath_toilet", "bath_basin"):
            raise _AttemptFailed(f"bathroom fixture {part} does not fit")
    if not any(item.slot == "bath_toilet" for item in placer.placed):
        raise _AttemptFailed("bathroom got no toilet")
    # a windowless bathroom is lit by its door alone and renders black
    _add_ceiling_lamp(placer, "ceiling_lamp_bath")


def _table_spots(table: Item, chair_dims: Sequence[float]):
    """Chair positions + yaws around a table, facing it (world coordinates)."""
    theta = math.radians(table.rot_z_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    half_l = table.dims[0] / 2.0
    half_d = table.dims[1] / 2.0
    spots = []
    for sign in (-1.0, 1.0):
        for offset in (-0.36, 0.36):
            v = sign * (half_d + 0.06 + chair_dims[1] / 2.0)
            rot = table.rot_z_deg + (0.0 if sign > 0 else 180.0)
            spots.append((offset, v, rot))
        u = sign * (half_l + 0.06 + chair_dims[1] / 2.0)
        rot = table.rot_z_deg + (270.0 if sign > 0 else 90.0)
        spots.append((u, 0.0, rot))
    world = []
    for u, v, rot in spots:
        world.append((table.pos[0] + cos_t * u - sin_t * v,
                      table.pos[1] + sin_t * u + cos_t * v, rot % 360.0))
    return world


def furnish_dining(placer: Placer) -> None:
    """A dining table with chairs on its long sides and ends."""
    rng = placer.rng
    table_dims = fixtures_mod.dims("dining_table")
    table = _place_free(placer, "dining_table", "dining_table", _fixture("dining_table"),
                        _free_candidates(placer, table_dims, step=0.3), attempts=30)
    if table is None:
        table = _place_on_wall(placer, "dining_table", "dining_table", _fixture("dining_table"),
                               _fixture_walls(placer), window_penalty=False)
    if table is None:
        raise _AttemptFailed("no room for a dining table")
    chair_dims = fixtures_mod.dims("dining_chair")
    chairs = 0
    for pos_x, pos_y, rot in _table_spots(table, chair_dims):
        item = placer.make("dining_chair", f"dining_chair_{chairs}", _fixture("dining_chair"),
                           (pos_x, pos_y), rot, checks_mod.FREE_STAND)
        if item is None or not placer.fits(item, checks_mod.WALL_GAP_MIN_M):
            continue
        added = placer.add(item, usage=0.40, allow=("dining_chair",), behind=True)
        if added is not None:
            chairs += 1
    if chairs < 2:
        raise _AttemptFailed(f"only {chairs} dining chairs fit")
    _add_ceiling_lamp(placer, "ceiling_lamp_dining")


def furnish_entry(placer: Placer) -> None:
    """Shoe cabinet (and a rug) in the entrance hall."""
    _place_on_wall(placer, "entry_shoe_cabinet", "shoe_cabinet", _fixture("entry_shoe_cabinet"),
                   _fixture_walls(placer), usage=fixtures_mod.usage_m("entry_shoe_cabinet"))
    rect = placer.rect
    rug_size = [min(1.6, rect.w - 0.8), min(0.9, rect.h - 0.8)]
    if min(rug_size) > 0.4 and placer.rng.random() < 0.5:
        item = placer.make("rug", "entry_rug", RUG_ASSET, rect.center, 0.0,
                           checks_mod.FLOOR_FLAT, size=rug_size)
        if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M, keep_doors=False):
            placer.add(item)
    _add_ceiling_lamp(placer, "ceiling_lamp_entry")


def furnish_storage(placer: Placer) -> None:
    """Shelving units for a storeroom."""
    walls = _fixture_walls(placer)
    _place_on_wall(placer, "storage_shelf", "storage_shelf", _fixture("storage_shelf"),
                   walls, usage=fixtures_mod.usage_m("storage_shelf"))
    if placer.rect.area >= 4.0:
        _place_on_wall(placer, "storage_shelf", "storage_shelf_2", _fixture("storage_shelf"),
                       list(reversed(walls)), usage=fixtures_mod.usage_m("storage_shelf"))
    _add_ceiling_lamp(placer, "ceiling_lamp_storage")


def furnish_laundry(placer: Placer) -> None:
    """Washing machine (and a dryer when the wall has room)."""
    walls = _fixture_walls(placer)
    washer = _place_on_wall(placer, "laundry_washer", "washer", _fixture("laundry_washer"),
                            walls, usage=fixtures_mod.usage_m("laundry_washer"))
    if washer is not None and placer.rect.area >= 3.6:
        wall = _facing_wall(placer, washer)
        along = washer.pos[0] if wall.side in ("N", "S") else washer.pos[1]
        offset = along + (washer.dims[0] + fixtures_mod.dims("laundry_washer")[0]) / 2.0 + 0.10
        item = placer.against_wall("laundry_washer", "dryer", _fixture("laundry_washer"), wall, offset)
        if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M):
            placer.add(item, usage=fixtures_mod.usage_m("laundry_washer"))
    _add_ceiling_lamp(placer, "ceiling_lamp_laundry")


def furnish_balcony(placer: Placer) -> None:
    """Plants on a balcony (it is the outdoor room, so no furniture to block it)."""
    rng = placer.rng
    for index in range(3):
        asset = placer.pick_asset("plant_small") or placer.pick_asset("plant_tall")
        if asset is None:
            break
        dims = placer.dims_of(asset)
        if dims is None:
            continue
        candidates = _corner_candidates(placer, dims, inset=0.35)
        _place_free(placer, "plant", f"plant_balcony_{index}", asset, candidates, attempts=4)
    if rng.random() < 0.5:
        basket_asset = placer.pick_asset("basket")
        if basket_asset:
            dims = placer.dims_of(basket_asset)
            _place_free(placer, "basket", "basket_balcony", basket_asset,
                        _corner_candidates(placer, dims, inset=0.35), attempts=4)


def furnish_corridor(placer: Placer) -> None:
    """A runner rug and a plant at the end: a corridor is circulation, not a room."""
    rng = placer.rng
    rect = placer.rect
    if rng.random() < 0.5:
        size = [max(0.5, rect.w - 0.5), max(0.4, rect.h - 0.35)] if rect.w >= rect.h else \
               [max(0.4, rect.w - 0.35), max(0.5, rect.h - 0.5)]
        item = placer.make("rug", "corridor_runner", RUG_ASSET, rect.center,
                           0.0 if size[0] >= size[1] else 90.0, checks_mod.FLOOR_FLAT, size=size)
        if item is not None and min(size) >= 0.4 and placer.fits(item, checks_mod.WALL_GAP_MIN_M, keep_doors=False):
            placer.add(item)
    asset = placer.pick_asset("plant_small") or placer.pick_asset("basket")
    if asset is not None and rng.random() < 0.6:
        dims = placer.dims_of(asset)
        _place_free(placer, "plant", "plant_corridor", asset, _corner_candidates(placer, dims),
                    attempts=4)
    # a windowless hall is lit by its doorways alone and renders black
    _add_ceiling_lamp(placer, "ceiling_lamp_corridor")


def furnish_plain(placer: Placer) -> None:
    """Dispatch of the more functional rooms; rooms without a handler stay bare."""
    room_type = placer.room.room_type
    if room_type == "kitchen":
        furnish_kitchen(placer)
    elif room_type == "bathroom":
        furnish_bathroom(placer)
    elif room_type == "study":
        furnish_study(placer)
    elif room_type == "dining":
        furnish_dining(placer)
    elif room_type == "entry":
        furnish_entry(placer)
    elif room_type == "storage":
        furnish_storage(placer)
    elif room_type == "laundry":
        furnish_laundry(placer)
    elif room_type == "balcony":
        furnish_balcony(placer)
    elif room_type == "corridor":
        furnish_corridor(placer)


def furnish_study(placer: Placer) -> None:
    """Desk (with its chair and its pull-out space), bookshelf and a lamp."""
    rng = placer.rng
    rect = placer.rect
    walls = _fixture_walls(placer, prefer_window=True)
    desk = _place_on_wall(placer, "study_desk", "desk", _fixture("study_desk"), walls,
                          usage=fixtures_mod.usage_m("study_desk"), allow=("study_chair",),
                          window_penalty=False)
    if desk is not None:
        chair_dims = fixtures_mod.dims("study_chair")
        facing = _facing_of(desk)
        gap = desk.dims[1] / 2.0 + 0.06 + chair_dims[1] / 2.0
        along = desk.pos[0] if abs(facing[1]) > 0 else desk.pos[1]
        for index, offset in enumerate((0.0, 0.35, -0.35)):
            pos = ((along + offset, desk.pos[1] + facing[1] * gap) if abs(facing[1]) > 0
                   else (desk.pos[0] + facing[0] * gap, along + offset))
            item = placer.make("study_chair", f"desk_chair_{index}", _fixture("study_chair"),
                               pos, (desk.rot_z_deg + 180.0) % 360.0, checks_mod.WALL_HUG)
            if item is not None and placer.fits(item, checks_mod.WALL_GAP_MIN_M) \
                    and placer.keeps_room_open(item) and placer.add(item) is not None:
                break
    shelf_asset = placer.pick_asset("bookshelf")
    if shelf_asset:
        avoid = [desk.aabb] if desk is not None else []
        others = [wall for wall in placer.walls.values() if desk is None or not _same_wall(wall, desk)]
        others.sort(key=lambda wall: (placer.wall_has_window(wall), -wall.length))
        _place_on_wall(placer, "bookshelf", "bookshelf_study", shelf_asset, others, avoid=avoid)
    plant_asset = placer.pick_asset("plant_small")
    if plant_asset and rng.random() < 0.6:
        dims = placer.dims_of(plant_asset)
        _place_free(placer, "plant", "plant_study", plant_asset, _corner_candidates(placer, dims),
                    attempts=4)
    _add_ceiling_lamp(placer, "ceiling_lamp_study")


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

    def _pixel_depth(self, origin, column: int, row: int, height: int, width: int = PANO_W) -> float:
        """Depth along the ERP ray of one pixel centre."""
        lon, lat = erp_lon_lat(column, row, width, height)
        return self.cast(tuple(origin), erp_ray_direction(lon, lat))

    def _window_median(self, origin, column: int, row: int, height: int, half: int = 2) -> float:
        """Median depth over the validator's 5x5 pixel window (its sampling rule).

        ``tools/validate_blender_agent_p0.py`` reads a rendered depth as the
        median of the 5x5 pixels around the projected door point, so the
        forecast has to do the same: one pixel can sit on the door jamb or under
        a lintel while the window still sees the opening, and then the forecast
        would disagree with the render by the width of one pixel.
        """
        depths = []
        for row_offset in range(-half, half + 1):
            r = max(0, min(height - 1, row + row_offset))
            for column_offset in range(-half, half + 1):
                c = (column + column_offset) % PANO_W
                depths.append(self._pixel_depth(origin, c, r, height))
        depths.sort()
        return depths[len(depths) // 2]

    def door_sight(self, door: DoorPlan, camera_pos: Sequence[float],
                   height: int = PANO_H) -> Optional[dict]:
        """Where the camera's ray towards this door goes: through it or into a wall.

        The door's mid-height point is projected into the panorama with the
        inverse camera rotation (:func:`erp_pixel_of_direction`), the ERP ray of
        that pixel is cast through the analytic shell, and the hit is compared
        with the distance to the door plane: a sight line that passes through
        the opening arrives at or behind the door plane (the tunnel is a hole in
        the wall box), one that is blocked by the wall, a lintel or furniture
        arrives in front of it.  This is exactly what
        ``tools/validate_blender_agent_p0.py``'s ``door_visibility_check``
        measures on the render, on the same pixel and with the same arithmetic.
        """
        if door.key() not in self.door_segments:
            return None
        point = (door.center[0], door.center[1], min(DOOR_H / 2.0, DOOR_H - 0.10))
        offset = (point[0] - camera_pos[0], point[1] - camera_pos[1], point[2] - camera_pos[2])
        expected = math.sqrt(sum(value * value for value in offset))
        if expected < 1e-6:
            return None
        column, row = erp_pixel_of_direction(offset, height=height)
        return {"door": door.key(),
                "column": column,
                "row": row,
                "expected_m": expected,
                "depth_m": self._pixel_depth(camera_pos, column, row, height),
                "visible": None}

    def ray_for_pixel(self, door: DoorPlan, camera_pos: Sequence[float],
                      height: int = PANO_H) -> Optional[dict]:
        """The validator's door visibility check, pre-computed on the analytic shell.

        Same projection (:func:`erp_pixel_of_direction`), same pixel window
        (:meth:`_window_median`) and the same "the ray arrives behind the door
        plane" rule as ``tools/validate_blender_agent_p0.py``, so the layout
        report can say what the render will show.  The old version reproduced the
        validator's *mirrored* column arithmetic instead, which only predicted
        the render for plans whose doors and cameras shared one Y line.
        """
        sight = self.door_sight(door, camera_pos, height)
        if sight is None:
            return None
        depth = self._window_median(camera_pos, sight["column"], sight["row"], height)
        visible = bool(depth >= sight["expected_m"] - CAMERA_DOOR_SIGHT_TOLERANCE_M)
        sight["depth_m"] = depth
        sight["visible"] = visible
        sight["ratio"] = (depth / sight["expected_m"]) if sight["expected_m"] > 0 else None
        return sight

def door_forecast(model: ShellModel, door: DoorPlan, camera_pos: Sequence[float]) -> Optional[dict]:
    """The validator's door visibility check, pre-computed on the analytic shell."""
    return model.ray_for_pixel(door, tuple(camera_pos))


def door_sight_line(model: ShellModel, door: DoorPlan, camera_pos: Sequence[float]) -> Optional[bool]:
    """Cheap door sight check for the camera search: centre ray only."""
    sight = model.door_sight(door, tuple(camera_pos))
    if sight is None:
        return None
    return sight["depth_m"] >= sight["expected_m"] - CAMERA_DOOR_SIGHT_TOLERANCE_M


def _walk_z(item: Item) -> Tuple[float, float]:
    """The Z range ``checks.ItemBox`` sees: a ceiling item is lifted to the top.

    ``Item.z_range`` is the *authored* range, and a ``mount: "ceiling"`` piece
    keeps ``z = None`` there; ``checks.item_boxes`` lifts it so its top touches
    the ceiling, and the generator's blocking tests have to do the same or a
    pendant lamp counts as furniture standing on the floor.
    """
    if item.placement == checks_mod.CEILING:
        z0 = WALL_HEIGHT - item.dims[2]
        return (z0, z0 + item.dims[2])
    return item.z_range


def _blocks_walking(item: Item) -> bool:
    """Mirror of ``checks.ItemBox.blocks_walking``: does the item block walkers?"""
    if item.placement == checks_mod.FLOOR_FLAT:
        return False
    z0, z1 = _walk_z(item)
    return z0 < checks_mod.WALK_BAND_M[1] and z1 > checks_mod.WALK_BAND_M[0]


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
                   step: float = CAMERA_GRID_M) -> List[Tuple[float, float]]:
    """Candidate stands: the room's walkable grid, sampled every ``CAMERA_GRID_M``.

    The whole walkable area is sampled, not a band around a common door line: a
    stand has to sit on a free cell the room's doorway BFS reaches (so the
    circulation check passes), keep ``CAMERA_INSET_M`` off the wall centre lines
    and ``CAMERA_MIN_GAP_TO_FURNITURE_M`` off every blocking footprint, and what
    it sees then decides where it goes (see :func:`place_cameras`).
    """
    rect = plan.rect
    blockers = [item for item in items if _blocks_camera(item)]
    stands: List[Tuple[float, float]] = []
    y = rect.y0 + CAMERA_INSET_M
    while y <= rect.y1 - CAMERA_INSET_M + 1e-9:
        x = rect.x0 + CAMERA_INSET_M
        while x <= rect.x1 - CAMERA_INSET_M + 1e-9:
            point = (round(x, 3), round(y, 3))
            if _walkable(grid, point[0], point[1], reachable=True):
                box = (point[0], point[1], point[0], point[1])
                if checks_mod.wall_gap(box, rect.polygon()) >= CAMERA_INSET_M - 1e-9 and \
                        all(checks_mod.rect_distance(box, item.aabb) >= CAMERA_MIN_GAP_TO_FURNITURE_M - 1e-9
                            for item in blockers):
                    stands.append(point)
            x += step
        y += step
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


def _forecast_entry(plan: RoomPlan, pick: dict, model: ShellModel) -> dict:
    """One ``door_visual_forecast`` row: what the stand was picked for, plus the
    pre-computed visibility of every doorway of the room."""
    view = pick["view"]
    clearance = view["furniture_clearance"]
    doors = []
    for door in pick["doors"]:
        sight = model.ray_for_pixel(door, (pick["pos"][0], pick["pos"][1], CAMERA_Z_M))
        if sight is None:
            continue
        doors.append({
            "between": [door.a, door.b],
            "column": sight["column"],
            "row": sight["row"],
            "door_distance_m": round(sight["expected_m"], 3),
            "rendered_depth_m": round(sight["depth_m"], 3),
            "ratio": None if sight["ratio"] is None else round(sight["ratio"], 3),
            "visible": bool(sight["visible"]),
        })
    seen = sum(1 for door in doors if door["visible"])
    return {
        "room": plan.room_id,
        "score": round(view["score"], 3),
        "doors": doors,
        "doors_visible": seen,
        "doors_checked": len(doors),
        "visible_fraction": round(seen / len(doors), 3) if doors else None,
        "door_sight_score": round(pick["sight"], 3),
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
                  want_second: bool) -> Tuple[List[dict], Optional[DoorPlan], dict]:
    """Pick 1-2 cameras per room by sampling the walkable area.

    Every room contributes at least one camera.  Stands come from the room's
    *walkable grid* (the free-cell lattice of :func:`_walkable_grid`, furniture
    footprints dilated as in :mod:`blender_agent.checks`) and are scored by what
    they see: the share of the walkable floor in unobstructed view, how much of
    the furniture is visible, the clearance to walls and furniture, the distance
    to the doorways and whether a doorway is itself on a clear sight line.  The
    old rule -- the point closest to the room centre -- put a camera in the
    middle of the room or in a corner behind the sofa; the view score picks the
    stand that shows the room.

    Nothing here is tied to a row any more.  The previous design kept stands in
    a band around one *common door line* and rejected every stand whose
    pre-computed door ratio fell below ``DOOR_FORECAST_MIN_RATIO``: that mirror
    -column forecast only reached a doorway from that band (see
    :meth:`ShellModel.ray_for_pixel`), so a 2-D plan had no usable stand at all.
    Door visibility is now a term in the score, not a filter: a room whose
    doorways are all hidden behind furniture still gets its best stand.

    The second camera in the living room has to stand
    ``CAMERA_SECOND_MIN_GAP_M`` away from the first and look at the room from a
    different bearing (``CAMERA_SECOND_MIN_ANGLE_DEG``), so a two-camera room is
    really seen from two sides.
    """
    items_by_room: Dict[str, List[Item]] = {plan.room_id: [] for plan in plans}
    for item in items:
        items_by_room[item.room].append(item)
    contexts: Dict[str, dict] = {}
    for plan in plans:
        context = _room_camera_context(plan, items_by_room[plan.room_id], doors)
        context["stands"] = _camera_stands(plan, items_by_room[plan.room_id], context["grid"])
        context["doors"] = [door for door in doors if plan.room_id in (door.a, door.b)]
        if not context["stands"]:
            raise _AttemptFailed(f"no walkable camera stand in room {plan.room_id}")
        contexts[plan.room_id] = context

    views: Dict[Tuple[str, float, float], dict] = {}
    sights: Dict[Tuple[str, float, float], float] = {}

    def view_of(plan: RoomPlan, pos: Sequence[float]) -> dict:
        key = (plan.room_id, pos[0], pos[1])
        if key not in views:
            views[key] = _camera_view(model, plan, contexts[plan.room_id], pos)
        return views[key]

    def sight_of(plan: RoomPlan, pos: Sequence[float]) -> float:
        """Share of the room's doorways this stand has a clear sight line to."""
        key = (plan.room_id, pos[0], pos[1])
        if key not in sights:
            room_doors = contexts[plan.room_id]["doors"]
            seen = sum(1 for door in room_doors
                       if door_sight_line(model, door, (pos[0], pos[1], CAMERA_Z_M)))
            sights[key] = (seen / len(room_doors)) if room_doors else 0.0
        return sights[key]

    def stand_value(plan: RoomPlan, pos: Sequence[float]) -> float:
        return view_of(plan, pos)["score"] + CAMERA_DOOR_SIGHT_WEIGHT * sight_of(plan, pos)

    def best_stand(plan: RoomPlan, candidates, value_fn) -> Optional[dict]:
        best = None
        for pos in candidates:
            value = value_fn(plan, pos)
            if value is None:
                continue
            if best is None or value > best["value"] + 1e-9:
                best = {"pos": pos, "view": view_of(plan, pos), "sight": sight_of(plan, pos),
                        "doors": contexts[plan.room_id]["doors"], "value": value}
        return best

    cameras: List[dict] = []
    camera_forecast: Dict[str, dict] = {}
    picks: Dict[str, dict] = {}
    index = 0
    for plan in plans:
        pick = best_stand(plan, contexts[plan.room_id]["stands"], stand_value)
        if pick is None:
            raise _AttemptFailed(f"no camera stand in room {plan.room_id}")
        picks[plan.room_id] = pick
        name = f"{index:04d}"
        index += 1
        cameras.append({"name": name, "room": plan.room_id,
                        "pos": [round(pick["pos"][0], 3), round(pick["pos"][1], 3), CAMERA_Z_M]})
        camera_forecast[name] = _forecast_entry(plan, pick, model)

    if want_second:
        living = next((plan for plan in plans if plan.room_type == "living_room"), None)
        if living is not None:
            first = picks[living.room_id]

            def second(plan: RoomPlan, min_angle: float) -> Optional[dict]:
                def value(plan: RoomPlan, pos: Sequence[float]):
                    gap = math.dist(pos, first["pos"])
                    if gap < CAMERA_SECOND_MIN_GAP_M - 1e-9:
                        return None
                    view = view_of(plan, pos)
                    angle = _bearing_angle(first["view"]["bearing"], view["bearing"])
                    if angle < min_angle - 1e-9:
                        return None
                    return (view["score"] + CAMERA_SECOND_GAP_WEIGHT * min(gap, 3.0)
                            + CAMERA_SECOND_ANGLE_WEIGHT * angle / 180.0
                            + CAMERA_DOOR_SIGHT_WEIGHT * sight_of(plan, pos)) * 1.0

                best = best_stand(plan, contexts[plan.room_id]["stands"], value)
                if best is not None:
                    best["gap"] = math.dist(best["pos"], first["pos"])
                    best["angle"] = _bearing_angle(first["view"]["bearing"], best["view"]["bearing"])
                return best

            # a second camera is only worth it if it really looks at the room from
            # another side; where the furniture leaves no stand that far around,
            # the best remaining stand still wins (the report keeps its angle)
            best = second(living, CAMERA_SECOND_MIN_ANGLE_DEG) or second(living, 0.0)
            if best is not None:
                name = f"{index:04d}"
                cameras.append({"name": name, "room": living.room_id,
                                "pos": [round(best["pos"][0], 3), round(best["pos"][1], 3), CAMERA_Z_M]})
                camera_forecast[name] = _second_camera_entry(
                    _forecast_entry(living, best, model), best)
    return cameras, _most_visible_opening(doors, cameras, camera_forecast), camera_forecast


def _most_visible_opening(doors: Sequence[DoorPlan], cameras: Sequence[dict],
                          forecast: Dict[str, dict]) -> Optional[DoorPlan]:
    """The opening the most cameras see through (a report label, not a rule)."""
    counts = {door.key(): 0 for door in doors}
    for camera in cameras:
        for sight in forecast.get(camera["name"], {}).get("doors", []):
            key = "|".join(sight["between"])
            if sight["visible"] and key in counts:
                counts[key] += 1
    if not counts:
        return None
    best_key = max(counts, key=lambda key: counts[key])
    return next(door for door in doors if door.key() == best_key)


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
            "axis-aligned rectangular rooms (one row, or a topology family: 2x2 block, L,",
            "central living, corridors, loop), doors from a semantic spanning tree over the",
            "shared walls, windows on exterior wall spans allocated by room function, furniture",
            "from the asset index plus the builtin: fixtures of blender_agent/fixtures.py with",
            "rot_z_deg on the 90 degree grid (front faces -Y at 0, see assets.py).  See the",
            "layout report for the per-camera door visibility forecast the validator measures.",
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
                   attempt: int, seed: int = 0, shape: str = "row") -> Tuple[dict, dict]:
    if shape == "auto":
        room_count, shape = _pick_auto_shape(rng, room_count)
    family = ROOM_FAMILIES.get(shape)
    if family is not None:
        count = family.count if room_count is None else int(room_count)
        if family.count != count:
            raise _AttemptFailed(f"layout '{shape}' needs {family.count} rooms, not {count}")
    else:
        count = room_count or rng.choice([2, 3, 3, 4, 4, 4, 5, 5])
        count = int(_clamp(count, 2, 5))
    if family is not None:
        plans = build_family(rng, family)
        doors = plan_doors_graph(rng, plans, extra_doors=family.extra_doors)
        semantics = door_semantics(plans, doors)
        if not semantics["pass"]:
            raise _AttemptFailed(f"topology {shape}: " + "; ".join(semantics["violations"][:2]))
    else:
        types = choose_room_types(rng, count)
        if shape == "block":
            plans = plan_rooms_block(rng, types)
            doors = plan_doors(rng, plans, chain=plans)
        else:
            plans = plan_rooms(rng, types)
            doors = plan_doors(rng, plans)
        semantics = door_semantics(plans, doors)
    windows = plan_windows(rng, plans)
    items, slot_report = furnish_rooms(rng, plans, doors, windows, index)
    rooms_by_id = {plan.room_id: plan.rect for plan in plans}
    model = ShellModel(rooms_by_id, doors, items)
    want_second = count <= 3
    cameras, opening0, forecast = place_cameras(rng, plans, doors, items, model, want_second)
    order = ([opening0] if opening0 is not None else []) + [door for door in doors if door is not opening0]
    spec = _spec_dict(seed, plans, doors, windows, cameras, items, order)
    spec["_comment"].append(f"rooms={count} attempt={attempt} furniture={len(items)} shape={shape}")
    report = {
        "shape": shape,
        "family": shape if family is not None else None,
        "door_semantics": semantics,
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
        "opening_0": opening0.key() if opening0 is not None else None,
        "door_visual_forecast": forecast,
    }
    return spec, report


def generate(seed: int, room_count: Optional[int] = None,
             assets_index: Optional[Dict[str, dict]] = None,
             attempts: int = MAX_LAYOUT_ATTEMPTS, verbose: bool = True,
             shape: str = "row") -> Scene:
    """Generate one layout; returns the spec dict plus the layout report.

    ``shape`` is ``"row"`` (the default single row), ``"block"`` (the same chain
    of rooms wrapped into two rows, see :func:`plan_rooms_block`), ``"auto"``
    (the seed draws the room count and one of :data:`ROOM_FAMILIES`), or any
    family name -- ``compact_2x2``, ``l_shape``, ``central_living``,
    ``corridor_single``, ``corridor_double``, ``loop``.
    """
    if shape not in ("row", "block", "auto") and shape not in ROOM_FAMILIES:
        raise ValueError(f"unknown layout shape '{shape}' (row, block, auto or a topology family)")
    if shape in ROOM_FAMILIES and room_count is not None and \
            ROOM_FAMILIES[shape].count != room_count:
        raise ValueError(f"layout '{shape}' has {ROOM_FAMILIES[shape].count} rooms, "
                         f"--rooms {room_count} does not match")
    index = assets_index if assets_index is not None else checks_mod.load_assets_index()
    started = time.time()
    last_error = "unknown"
    for attempt in range(attempts):
        rng = random.Random(seed * 1000003 + attempt)
        try:
            spec_dict, report = _build_attempt(rng, room_count, index, attempt, seed, shape)
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
            "shape": shape,
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
    parser.add_argument("--rooms", type=int, default=None,
                        help="2..7, default: drawn from the seed")
    parser.add_argument("--shape", default=os.environ.get("PANOWORLD_LAYOUT_SHAPE", "row"),
                        choices=("row", "block", "auto") + tuple(sorted(ROOM_FAMILIES)),
                        help="room arrangement: 'row' (one row along +X, default), 'block' (the same "
                             "chain of rooms wrapped into two rows), 'auto' (the seed draws the room "
                             "count and a topology family) or a family name: "
                             + ", ".join(sorted(ROOM_FAMILIES))
                             + " -- also settable with PANOWORLD_LAYOUT_SHAPE, which is what "
                               "scripts/layout_gen_run.sh passes through")
    parser.add_argument("--assets-index", default=DEFAULT_ASSETS_INDEX)
    parser.add_argument("--report", default=None, help="layout report (default: next to --out)")
    parser.add_argument("--attempts", type=int, default=MAX_LAYOUT_ATTEMPTS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.rooms is not None and not 2 <= args.rooms <= 7:
        parser.error("--rooms must be between 2 and 7")

    index = checks_mod.load_assets_index(args.assets_index)
    try:
        scene = generate(args.seed, args.rooms, index, attempts=args.attempts,
                         verbose=not args.quiet, shape=args.shape)
    except ValueError as error:                 # a room count or shape the CLI cannot honour
        print(f"[layout_gen] FAILED: {error}", file=sys.stderr)
        return 2
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
