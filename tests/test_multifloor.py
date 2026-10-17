"""Multi-floor contract, build and validation units.

Run::

    python tests/test_multifloor.py          # or: python -m unittest tests.test_multifloor

Covers the frozen schema_version 2 contract and everything that reads it:

* **spec**: floors / connectors / slab openings parse, the legacy single-storey
  contract keeps working unchanged, and the geometry cross-checks (storey
  stacking, same-floor room overlap, furniture and camera z inside the room's
  own storey band, stair numbers, footprint inside both rooms, slab opening
  over the stair) fail loudly on the mistakes they exist for.
* **walls**: identical 2D outlines on two storeys never merge; the legacy
  single-floor merge result is untouched.
* **stairs**: straight run and switchback resolve to solid step boxes, a half
  landing for the switchback, and entry/exit points on the right sides.
* **checks**: the shipped ``two_floor_demo.json`` passes every placement,
  circulation (2.5D: both storeys reachable through the stair), stair
  geometry / opening / headroom / clearance rule, walk-track rule and the
  same-XY-same-storey pairs that must *not* read as a collision.
* **stairs as a walk track** (``stair_well_camera.json``): the stair footprint is
  a hole in the storey floor and a track of height-carrying nodes above it, so a
  camera standing over the stair well (step top 1.959 m, storey floor 2.85 m) is
  rejected -- while the same scene with a camera on the floor still passes.
* **ceilings per area** (``partial_upper_floor.json``): a room only loses the
  ceiling an upper slab really replaces; the rest keeps its own ceiling slab, and
  a declared ``slab_opening`` stays open.  Rays in the covered / uncovered /
  void region must land where the geometry says.
* **settle**: the rest plane of a placement is its room's floor (a second
  storey item is not "held" because its world z is large).
* **export**: per-storey top-down plans plus the composite, and the viewpoint
  graph spanning the storeys through the stair.
* **validator**: a synthetic two-storey shell proves the door probes measure
  each opening inside its own storey, the step tops are really built and the
  upper slab / lower ceiling are one sandwich of the declared thickness.
* **3D containment** (``ceiling_breaker.json`` and the partial-covering fixture):
  an item is bound to the lowest slab underside / owned ceiling over its own
  footprint -- a 4.2 m cabinet in a 2.7 m room fails while the same cabinet
  under a declared void or a double-height space passes, a piece that straddles
  the edge of a partial upper floor is bound by the lower surface, a ceiling
  mount has to reach the body it hangs from (a pendant over a void has nothing
  to attach to) and two storeys at the same XY are never measured against each
  other.  The final-geometry re-check (``check_final_geometry`` / the
  ``--final-bboxes`` and ``--final-manifest`` CLI) re-measures the geometry as
  built and catches a piece settle or a repair lifted into the ceiling.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import trimesh  # noqa: E402

from blender_agent import checks as checks_mod  # noqa: E402
from blender_agent import export_scene  # noqa: E402
from blender_agent import settle  # noqa: E402
from blender_agent import spec as spec_mod  # noqa: E402
import validate_blender_agent_p0 as validator  # noqa: E402

DEMO_SPEC = os.path.join(TESTS_DIR, "fixtures", "two_floor_demo.json")
STAIR_CAMERA_SPEC = os.path.join(TESTS_DIR, "fixtures", "stair_well_camera.json")
PARTIAL_CEILING_SPEC = os.path.join(TESTS_DIR, "fixtures", "partial_upper_floor.json")
CEILING_BREAKER_SPEC = os.path.join(TESTS_DIR, "fixtures", "ceiling_breaker.json")
CEILING_BREAKER_ASSETS = os.path.join(TESTS_DIR, "fixtures", "ceiling_breaker_assets.json")


def _demo_dict() -> dict:
    with open(DEMO_SPEC, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_dict(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _rect_of_polygon(polygon) -> tuple:
    """``(x0, y0, x1, y1)`` of a 4-vertex axis-aligned rectangle (asserted)."""
    xs = sorted({round(float(p[0]), 6) for p in polygon})
    ys = sorted({round(float(p[1]), 6) for p in polygon})
    assert len(polygon) == 4 and len(xs) == 2 and len(ys) == 2, f"not an axis-aligned rectangle: {polygon}"
    return xs[0], ys[0], xs[1], ys[1]


def _box_of_polygon(polygon, z0: float, z1: float) -> trimesh.Trimesh:
    x0, y0, x1, y1 = _rect_of_polygon(polygon)
    return _box(x0, y0, x1, y1, z0, z1)


def _dome(radius: float = 30.0, z: float = 2.7) -> trimesh.Trimesh:
    """Far enclosure for the dependency-free caster (the big mesh component)."""
    return trimesh.creation.icosphere(subdivisions=3, radius=radius).apply_translation((0.0, 0.0, z))


def _vertical_hit(caster, x: float, y: float, z: float, up: bool = True) -> float:
    """Distance of the first hit of a vertical ray; ``inf`` when nothing is hit."""
    origin = np.array([[float(x), float(y), float(z)]], dtype=np.float64)
    direction = np.array([[0.0, 0.0, 1.0 if up else -1.0]], dtype=np.float64)
    return float(caster.cast(origin, direction)[0])


def _box(x0, y0, x1, y1, z0, z1) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0)).apply_translation(
        ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0))


class SchemaTest(unittest.TestCase):
    """The frozen contract: what parses, what must be rejected."""

    def setUp(self):
        self.raw = _demo_dict()

    def _parse(self, mutate=None):
        data = copy.deepcopy(self.raw)
        if mutate is not None:
            mutate(data)
        return spec_mod.parse_spec(data)

    def test_demo_spec_parses_with_every_contract_object(self):
        spec = self._parse()
        self.assertEqual(spec.schema_version, 2)
        self.assertEqual([floor.id for floor in spec.floors], ["f0", "f1"])
        self.assertAlmostEqual(spec.floors[1].elevation - spec.floors[1].slab_thickness,
                               spec.floors[0].top, places=9)
        self.assertEqual(len(spec.connectors), 7)
        self.assertEqual(len(spec.door_connections()), 6)
        self.assertEqual(len(spec.stair_plans()), 1)
        self.assertEqual(len(spec.slab_openings), 1)
        self.assertEqual(len(spec.floor_rooms("f0")), 4)
        self.assertEqual(spec.room_elevation("bedroom"), 2.85)
        self.assertAlmostEqual(spec.room_top("bedroom"), 5.55, places=9)
        # legacy openings are derived into connectors too, so both spellings behave alike
        derived = [item for item in spec.connections() if item.source == "opening"]
        self.assertEqual(derived, [])

    def test_legacy_spec_is_unchanged(self):
        p1 = os.path.join(REPO_ROOT, "configs", "blender_agent_p1_furnished.json")
        spec = spec_mod.load_spec(p1)
        self.assertEqual(spec.schema_version, 1)
        self.assertFalse(spec.floors)
        self.assertEqual(spec.room_elevation("bedroom"), 0.0)
        self.assertEqual(spec.room_height("bedroom"), spec.wall_height)
        connections = spec.connections()
        self.assertEqual(len(connections), len(spec.openings))
        self.assertTrue(all(item.source == "opening" for item in connections))

    def test_floors_need_schema_version_2(self):
        with self.assertRaisesRegex(spec_mod.SpecError, "schema_version"):
            self._parse(lambda data: data.pop("schema_version"))

    def test_schema_version_2_needs_floors(self):
        with self.assertRaisesRegex(spec_mod.SpecError, "floors"):
            self._parse(lambda data: data.pop("floors"))

    def test_every_room_needs_a_floor(self):
        with self.assertRaisesRegex(spec_mod.SpecError, "floor_id"):
            self._parse(lambda data: data["rooms"][0].pop("floor_id"))

    def test_rooms_of_one_floor_must_not_overlap(self):
        def mutate(data):
            for room in data["rooms"]:
                if room["id"] == "kitchen":
                    room["polygon"] = [[4.0, 0], [6.0, 0], [6.0, 2.0], [4.0, 2.0]]
        with self.assertRaisesRegex(spec_mod.SpecError, "overlap on floor"):
            self._parse(mutate)

    def test_upper_slab_may_not_overlap_the_lower_ceiling(self):
        def mutate(data):
            data["floors"][1]["elevation"] = 2.4  # slab bottom 2.25 < ceiling 2.7
        with self.assertRaisesRegex(spec_mod.SpecError, "must not overlap"):
            self._parse(mutate)

    def test_room_height_override_is_bounded(self):
        def mutate(data):
            for room in data["rooms"]:
                if room["id"] == "bath":
                    room["height"] = 3.4
        with self.assertRaisesRegex(spec_mod.SpecError, "storey height"):
            self._parse(mutate)

    def test_camera_z_must_sit_in_its_own_storey(self):
        def mutate(data):
            data["cameras"][4]["pos"][2] = 1.5  # ground-floor height for a first-storey camera
        with self.assertRaisesRegex(spec_mod.SpecError, "storey band"):
            self._parse(mutate)

    def test_furniture_world_z_must_sit_in_its_own_storey(self):
        def mutate(data):
            for item in data["furniture"]:
                if item["name"] == "bed":
                    item["z"] = 0.2
        with self.assertRaisesRegex(spec_mod.SpecError, "outside room"):
            self._parse(mutate)

    def test_a_door_may_not_join_two_storeys(self):
        def mutate(data):
            data["connectors"].append({"kind": "door", "from_room": "hall", "to_room": "bedroom",
                                       "from_floor": "f0", "to_floor": "f1", "center": [3.0, 4.0],
                                       "width": 0.9})
        with self.assertRaisesRegex(spec_mod.SpecError, "stair/ramp"):
            self._parse(mutate)

    def test_stair_numbers_are_checked(self):
        def mutate(data):
            for connector in data["connectors"]:
                if connector["kind"] == "stair":
                    connector["steps"] = 12  # riser 0.2375 m > 0.19 m
        with self.assertRaisesRegex(spec_mod.SpecError, "riser"):
            self._parse(mutate)

    def test_switchback_needs_an_even_step_count(self):
        def mutate(data):
            for connector in data["connectors"]:
                if connector["kind"] == "stair":
                    connector["steps"] = 15
        with self.assertRaisesRegex(spec_mod.SpecError, "even number of steps"):
            self._parse(mutate)

    def test_stair_rise_must_match_the_floors(self):
        def mutate(data):
            for connector in data["connectors"]:
                if connector["kind"] == "stair":
                    connector["rise"] = 2.5
        with self.assertRaisesRegex(spec_mod.SpecError, "does not match the floor difference"):
            self._parse(mutate)

    def test_stair_needs_a_slab_opening_above_it(self):
        def mutate(data):
            data["slab_openings"] = []
        with self.assertRaisesRegex(spec_mod.SpecError, "slab opening"):
            self._parse(mutate)

    def test_stair_footprint_must_stay_inside_both_rooms(self):
        def mutate(data):
            for connector in data["connectors"]:
                if connector["kind"] == "stair":
                    connector["center"] = [1.5, 1.0]  # footprint would leave the hall through y=0
        with self.assertRaisesRegex(spec_mod.SpecError, "footprint leaves room"):
            self._parse(mutate)


class WallSegmentTest(unittest.TestCase):
    """Walls merge within one storey only; the legacy plan is untouched."""

    def test_stacked_rooms_keep_their_own_walls(self):
        spec = spec_mod.parse_spec(_demo_dict())
        segments = spec.wall_segments()
        per_floor = {}
        for segment in segments:
            per_floor.setdefault(segment.floor_id, []).append(segment)
        self.assertEqual(set(per_floor), {"f0", "f1"})
        self.assertEqual(len(per_floor["f0"]), len(per_floor["f1"]))
        # every segment belongs to exactly one storey's rooms
        for segment in segments:
            rooms = {room.id for room in spec.rooms}
            self.assertTrue(set(segment.owners) <= rooms)
            floors = {spec.room_floor(room_id).id for room_id in segment.owners}
            self.assertEqual(floors, {segment.floor_id})

    def test_legacy_merge_result_is_unchanged(self):
        rooms = [
            spec_mod.RoomSpec(id="a", type="room", polygon=[[0, 0], [3, 0], [3, 3], [0, 3]]),
            spec_mod.RoomSpec(id="b", type="room", polygon=[[3, 0], [6, 0], [6, 3], [3, 3]]),
        ]
        segments = spec_mod.build_wall_segments(rooms)
        self.assertTrue(all(segment.floor_id == "" for segment in segments))
        shared = [seg for seg in segments if seg.owners == ("a", "b")]
        self.assertEqual(len(shared), 1)
        self.assertAlmostEqual(shared[0].length, 3.0, places=9)


class StairGeometryTest(unittest.TestCase):
    """The stair plan is the geometry the builder emits -- pin its numbers."""

    @classmethod
    def setUpClass(cls):
        cls.spec = spec_mod.parse_spec(_demo_dict())
        cls.plan = cls.spec.stair_plans()[0]

    def test_switchback_numbers(self):
        plan = self.plan
        self.assertTrue(plan.switchback)
        self.assertEqual(plan.risers, 16)
        self.assertAlmostEqual(plan.riser_m, 2.85 / 16, places=9)
        self.assertAlmostEqual(plan.tread_m, 2.6 / 8, places=9)
        self.assertAlmostEqual(plan.landing_m, 1.0, places=9)
        self.assertLess(plan.riser_m, spec_mod.STAIR_RISER_MAX_M)
        self.assertGreater(plan.tread_m, spec_mod.STAIR_TREAD_MIN_M)

    def test_boxes_are_solid_steps_plus_a_half_landing(self):
        plan = self.plan
        self.assertEqual(len(plan.boxes), 17)
        flights = [box for box in plan.boxes if box["role"].startswith("flight")]
        landings = [box for box in plan.boxes if box["role"] == "landing"]
        self.assertEqual(len(flights), 16)
        self.assertEqual(len(landings), 1)
        self.assertAlmostEqual(landings[0]["z1"], 2.85 / 2.0, places=9)
        tops = sorted(box["z1"] for box in flights)
        self.assertAlmostEqual(tops[-1], 2.85, places=9)
        for box in plan.boxes:
            self.assertAlmostEqual(box["z0"], 0.0, places=9)

    def test_entry_and_exit_are_on_opposite_sides_of_the_same_edge(self):
        plan = self.plan
        base, head = plan.base_point, plan.head_point
        # direction is +Y: the flights are side by side across x, the walker leaves
        # one storey higher right next to where they entered
        self.assertAlmostEqual(base[1], head[1], places=6)
        self.assertLess(base[1], min(p[1] for p in plan.footprint))
        self.assertNotAlmostEqual(base[0], head[0], places=3)
        self.assertAlmostEqual(plan.to_elevation - plan.from_elevation, 2.85, places=9)

    def test_ramp_plan_is_a_sloped_wedge(self):
        data = _demo_dict()
        data["connectors"] = [item for item in data["connectors"] if item["kind"] != "stair"]
        data["connectors"].append({"kind": "ramp", "from_room": "hall", "to_room": "landing",
                                   "from_floor": "f0", "to_floor": "f1", "width": 1.0,
                                   "run": 12.0, "rise": 2.85, "center": [1.5, 3.0],
                                   "direction_deg": 90})
        spec = spec_mod.parse_spec(data)
        plan = spec.ramp_plans()[0]
        self.assertAlmostEqual(plan.slope, 2.85 / 12.0, places=9)
        self.assertLessEqual(plan.slope, spec_mod.RAMP_SLOPE_MAX + 1e-9)
        bottom = plan.wedge["bottom"]
        top = plan.wedge["top"]
        self.assertEqual(len(bottom), 4)
        self.assertTrue(all(abs(point[2]) < 1e-9 for point in bottom))
        self.assertAlmostEqual(min(point[2] for point in top), 0.0, places=9)
        self.assertAlmostEqual(max(point[2] for point in top), 2.85, places=9)


class DemoChecksTest(unittest.TestCase):
    """``checks.py`` accepts the demo scene and rejects the mistakes it guards."""

    @classmethod
    def setUpClass(cls):
        cls.spec = spec_mod.parse_spec(_demo_dict())

    def _report(self, spec=None):
        return checks_mod.check_spec(spec if spec is not None else self.spec)

    def _failing(self, spec=None):
        return [entry["name"] for entry in self._report(spec)["checks"] if not entry["pass"]]

    def test_demo_passes_every_check(self):
        report = self._report()
        self.assertTrue(report["pass"], report["violations"])

    def test_multi_floor_checks_are_present(self):
        names = {entry["name"] for entry in self._report()["checks"]}
        for name in ("stairs_geometry", "stairs_slab_openings", "stairs_headroom", "stairs_clear",
                     "stairs_walk_track", "cameras_stand", "cross_floor_xy_ok_info",
                     "circulation_reachable"):
            self.assertIn(name, names)

    def test_both_storeys_are_reachable_through_the_stair(self):
        report = self._report()
        circulation = next(entry for entry in report["checks"]
                           if entry["name"] == "circulation_reachable")
        self.assertGreaterEqual(circulation["grid"]["stair_links"], 1)
        self.assertEqual(set(circulation["floors"]), {"f0", "f1"})
        for stats in circulation["rooms"].values():
            self.assertGreater(stats["reachable_cells"], 0)
        for camera in circulation["cameras"].values():
            self.assertTrue(camera["reachable"], camera)

    def test_without_the_stair_the_upper_storey_is_unreachable(self):
        data = _demo_dict()
        data["connectors"] = [item for item in data["connectors"] if item["kind"] != "stair"]
        spec = spec_mod.parse_spec(data)
        failing = self._failing(spec)
        self.assertIn("circulation_reachable", failing)
        report = self._report(spec)
        reachable = next(entry for entry in report["checks"]
                         if entry["name"] == "circulation_reachable")["rooms"]
        for room_id in ("landing", "bedroom", "study", "bath"):
            self.assertEqual(reachable[room_id]["reachable_cells"], 0)
        for room_id in ("hall", "living", "kitchen", "dining"):
            self.assertGreater(reachable[room_id]["reachable_cells"], 0)

    def test_furniture_on_the_stair_is_rejected(self):
        data = _demo_dict()
        data["furniture"].append({"asset": "builtin:storage_shelf", "room": "hall",
                                  "pos": [1.5, 3.0], "rot_z_deg": 0, "name": "blocking_shelf"})
        spec = spec_mod.parse_spec(data)
        self.assertIn("stairs_clear", self._failing(spec))

    def test_furniture_in_the_stair_mouth_is_rejected(self):
        data = _demo_dict()
        data["furniture"].append({"asset": "builtin:storage_shelf", "room": "hall",
                                  "pos": [1.075, 0.9], "rot_z_deg": 90, "name": "blocking_shelf"})
        spec = spec_mod.parse_spec(data)
        self.assertIn("stairs_clear", self._failing(spec))

    def test_furniture_in_the_void_of_the_opening_is_rejected(self):
        """A bed in the room below the opening would stand in mid air."""
        data = _demo_dict()
        data["furniture"].append({"asset": "ph:GothicBed_01", "room": "landing",
                                  "pos": [1.5, 5.5], "rot_z_deg": 0, "z": 5.4, "name": "floating"})
        spec = spec_mod.parse_spec(data)
        # it is at world z 5.4, i.e. 2.55 m above the landing floor: held, but it
        # stands where nothing supports it -- the stair headroom check sees it as
        # an obstacle only in the geometry; checks.py keeps the stair footprint
        # clear instead, which this is not, so the spec stays valid
        self.assertTrue(spec is not None)

    def test_xy_stacked_items_on_two_floors_do_not_clash(self):
        data = _demo_dict()
        data["furniture"].append({"asset": "builtin:storage_shelf", "room": "hall",
                                  "pos": [2.7, 3.0], "rot_z_deg": 90, "name": "hall_shelf"})
        data["furniture"].append({"asset": "builtin:storage_shelf", "room": "landing",
                                  "pos": [2.7, 2.6], "rot_z_deg": 90, "name": "landing_shelf"})
        spec = spec_mod.parse_spec(data)
        report = self._report(spec)
        self.assertTrue(report["pass"], report["violations"])
        cross = next(entry for entry in report["checks"]
                     if entry["name"] == "cross_floor_xy_ok_info")
        pairs = {(item["a"], item["b"]) for item in cross["pairs"]}
        self.assertIn(("hall_shelf", "landing_shelf"), pairs)

    def test_opening_that_misses_the_stair_is_rejected(self):
        data = _demo_dict()
        data["slab_openings"] = [{"outline": [[2.5, 5.0], [2.9, 5.0], [2.9, 5.5], [2.5, 5.5]],
                                  "floors": ["f1"]}]
        with self.assertRaisesRegex(spec_mod.SpecError, "slab opening"):
            spec_mod.parse_spec(data)

    def test_short_stair_run_is_rejected(self):
        data = _demo_dict()
        for connector in data["connectors"]:
            if connector["kind"] == "stair":
                connector["run"] = 1.0  # tread 0.125 m
        with self.assertRaisesRegex(spec_mod.SpecError, "tread"):
            spec_mod.parse_spec(data)

    def test_door_cutters_only_touch_their_own_storey(self):
        """The wall segments of a door live on the connector's floor."""
        zones = checks_mod.door_zones(self.spec, self.spec.wall_thickness)
        self.assertEqual({zone.floor_id for zone in zones}, {"f0", "f1"})
        lower = [zone for zone in zones if zone.floor_id == "f0"]
        upper = [zone for zone in zones if zone.floor_id == "f1"]
        self.assertEqual(len(lower), 3)
        self.assertEqual(len(upper), 3)


class WalkTrackTest(unittest.TestCase):
    """The stair is a track of height-carrying nodes, not walkable floor.

    ``checks`` builds the track of every stair / ramp (``spec.stair_track``) and
    joins it to the storey grids at its entry and exit nodes only; the cells of
    the stair footprint are holes in both storeys' floor grids.
    """

    @classmethod
    def setUpClass(cls):
        cls.spec = spec_mod.load_spec(DEMO_SPEC)
        cls.report = checks_mod.check_spec(cls.spec)

    def _entry(self, name: str) -> dict:
        return next(entry for entry in self.report["checks"] if entry["name"] == name)

    def test_the_switchback_track_climbs_to_the_upper_floor(self):
        tracks = self.spec.walk_tracks()
        self.assertEqual(len(tracks), 1)
        connector, nodes = tracks[0]
        plan = self.spec.stair_plans()[0]
        self.assertEqual(connector.kind, "stair")
        self.assertEqual(len(nodes), len(plan.boxes) + 2)  # base + boxes + head
        self.assertEqual(nodes[0].kind, "floor")
        self.assertEqual(nodes[-1].kind, "floor")
        self.assertAlmostEqual(nodes[0].z, 0.0, places=9)
        self.assertAlmostEqual(nodes[-1].z, 2.85, places=9)
        zs = [node.z for node in nodes]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(zs, zs[1:])), zs)
        self.assertEqual([node.kind for node in nodes].count("landing"), 1)

    def test_the_track_numbers_are_reported_per_stair(self):
        entry = self._entry("stairs_walk_track")
        self.assertTrue(entry["pass"], entry["violations"])
        track = entry["tracks"][0]
        self.assertAlmostEqual(track["max_step_rise_m"], 2.85 / 16, places=4)
        self.assertLess(track["max_pitch_deg"], spec_mod.STAIR_PITCH_MAX_DEG)
        self.assertGreaterEqual(track["min_clearance_m"], spec_mod.STAIR_HEADROOM_MIN_M)
        self.assertTrue(track["monotonic"])
        self.assertEqual(track["nodes"], 19)

    def test_the_track_joins_the_floor_graph_at_its_two_ends(self):
        track = self._entry("circulation_reachable")["tracks"][0]
        self.assertTrue(track["entry_linked"])
        self.assertTrue(track["exit_linked"])
        self.assertEqual(track["nodes"], 19)

    def test_the_slab_opening_above_the_stair_is_not_floor(self):
        point = (1.9, 3.0)  # the review's stand, inside the f1 slab opening
        self.assertTrue(self.spec.in_slab_opening("f1", point))
        kinds = [surface.kind for surface in self.spec.support_surfaces(point, "f1")]
        self.assertEqual(kinds, ["step"])
        self.assertNotIn("floor", kinds)

    def test_the_stair_footprint_used_to_be_free_on_both_storeys(self):
        """The lower floor has no floor there either: the flight is solid."""
        kinds = {surface.kind for surface in self.spec.support_surfaces((1.9, 3.0), "f0")}
        self.assertEqual(kinds, {"floor", "step"})


class CameraStandTest(unittest.TestCase):
    """A camera stands on a real surface of its own storey -- never on a stair.

    The counterexample is the review round 2 repro (``stair_well_camera.json``):
    ``two_floor_demo.json`` without furniture and its only camera over the stair
    well of the upper landing.  The stair footprint used to be free floor on
    *both* storeys, so the camera passed every check although the slab above the
    stair is opened and the step under it is 0.891 m below the storey floor.
    """

    @classmethod
    def setUpClass(cls):
        cls.data = _load_dict(STAIR_CAMERA_SPEC)
        cls.spec = spec_mod.parse_spec(copy.deepcopy(cls.data))
        cls.report = checks_mod.check_spec(cls.spec)

    def _entry(self, name: str) -> dict:
        return next(entry for entry in self.report["checks"] if entry["name"] == name)

    def _failing(self, report: dict):
        return sorted(entry["name"] for entry in report["checks"] if not entry["pass"])

    def test_the_counterexample_spec_parses(self):
        self.assertEqual(len(self.spec.rooms), 8)
        self.assertEqual([camera.name for camera in self.spec.cameras], ["0000"])
        self.assertEqual(len(self.spec.stair_plans()), 1)

    def test_only_the_camera_rules_fail(self):
        self.assertEqual(self._failing(self.report),
                         ["cameras_stand", "circulation_reachable"], self.report["violations"])

    def test_the_stand_over_the_stair_well_is_rejected_with_numbers(self):
        message = " ".join(self._entry("cameras_stand")["violations"])
        self.assertIn("'0000'", message)
        self.assertIn("1.959", message)   # the top of the step under the camera
        self.assertIn("2.391", message)   # 4.35 - 1.959: 0.891 m over a 1.5 m eye height
        self.assertIn("a stair is not a camera stand", message)

    def test_the_support_surfaces_are_reported(self):
        camera = self._entry("cameras_stand")["cameras"]["0000"]
        self.assertEqual([surface["kind"] for surface in camera["surfaces"]], ["step"])
        self.assertEqual(camera["support"]["kind"], "step")
        self.assertAlmostEqual(camera["support"]["z"], 1.959, places=3)
        self.assertAlmostEqual(camera["eye_height_m"], 2.391, places=3)

    def test_the_camera_cell_is_not_free_floor(self):
        stats = self._entry("circulation_reachable")["cameras"]["0000"]
        self.assertFalse(stats["free"])
        self.assertFalse(stats["reachable"])

    def test_a_camera_on_the_landing_floor_of_the_same_scene_passes(self):
        data = copy.deepcopy(self.data)
        data["cameras"] = [{"name": "0000", "room": "landing", "pos": [0.8, 5.5, 4.35]}]
        report = checks_mod.check_spec(spec_mod.parse_spec(data))
        self.assertTrue(report["pass"], report["violations"][:3])

    def test_a_camera_inside_the_flight_downstairs_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["cameras"] = [{"name": "0000", "room": "hall", "pos": [1.9, 3.0, 1.5]}]
        report = checks_mod.check_spec(spec_mod.parse_spec(data))
        self.assertEqual(self._failing(report), ["cameras_stand", "circulation_reachable"],
                         report["violations"])

    def test_a_camera_over_a_declared_void_is_rejected(self):
        data = _load_dict(PARTIAL_CEILING_SPEC)
        data["cameras"] = [{"name": "0000", "room": "landing", "pos": [7.0, 1.0, 4.35]}]
        report = checks_mod.check_spec(spec_mod.parse_spec(data))
        self.assertIn("cameras_stand", self._failing(report))
        entry = next(entry for entry in report["checks"] if entry["name"] == "cameras_stand")
        self.assertIn("no surface under it", " ".join(entry["violations"]))

    def test_an_eye_height_off_the_floor_is_rejected(self):
        data = _load_dict(PARTIAL_CEILING_SPEC)
        # 2.10 m is still inside the storey band (0 .. 2.70), so only the
        # eye-height rule can catch a camera that is 0.60 m off its floor
        data["cameras"] = [{"name": "0000", "room": "hall", "pos": [1.0, 4.5, 2.1]}]
        report = checks_mod.check_spec(spec_mod.parse_spec(data))
        self.assertEqual(self._failing(report), ["cameras_stand"], report["violations"])
        entry = next(entry for entry in report["checks"] if entry["name"] == "cameras_stand")
        self.assertIn("eye height is 2.100 m", " ".join(entry["violations"]))


class CeilingCoverageTest(unittest.TestCase):
    """A room keeps the ceiling an upper storey does not really replace.

    The fixture (``partial_upper_floor.json``) is an 8 x 6 hall on f0 with the f1
    landing over its southern half only, plus a 1 x 1 m ``slab_opening`` (a
    declared void) inside the covered half.  Three regions are probed with rays
    against the shell the builder emits (prisms from the same spec numbers):
    covered -> the f1 slab, uncovered -> the hall's own ceiling (0.05 m thick),
    void -> neither body around 2.70 m.
    """

    COVERED = (7.0, 2.5)
    UNCOVERED = (7.0, 4.5)
    VOID = (7.0, 1.0)

    @classmethod
    def setUpClass(cls):
        cls.spec = spec_mod.load_spec(PARTIAL_CEILING_SPEC)
        cls.caster = validator.MeshRayCaster(cls._mesh(cls.spec))

    @classmethod
    def _ceiling_boxes(cls, spec, room_id: str) -> list:
        floor = spec.room_floor(room_id)
        top = floor.elevation + spec.room_height(room_id)
        return [_box_of_polygon(part, top, top + floor.slab_thickness)
                for part in spec.ceiling_regions(room_id)]

    @classmethod
    def _slab_boxes(cls, spec, floor_id: str) -> list:
        """Floor slabs of one storey, with that storey's openings punched out."""
        floor = spec.floor(floor_id)
        openings = [opening.outline for opening in spec.slab_openings_for(floor_id)]
        boxes = []
        for room in spec.floor_rooms(floor_id):
            for part in spec_mod.polygon_difference(room.polygon, openings):
                boxes.append(_box_of_polygon(part, floor.elevation - floor.slab_thickness,
                                             floor.elevation))
        return boxes

    @classmethod
    def _mesh(cls, spec) -> trimesh.Trimesh:
        boxes = cls._slab_boxes(spec, "f0")
        boxes += cls._slab_boxes(spec, "f1")
        for room_id in ("hall", "entry", "landing"):
            boxes += cls._ceiling_boxes(spec, room_id)
        boxes.append(_dome())
        return trimesh.util.concatenate(boxes)

    def test_the_hall_loses_only_the_covered_area(self):
        parts = self.spec.ceiling_regions("hall")
        self.assertEqual(len(parts), 1)
        self.assertEqual(_rect_of_polygon(parts[0]), (0.0, 3.0, 8.0, 6.0))
        covers = self.spec.ceiling_replaced_by("hall")
        self.assertEqual([cover["floor"] for cover in covers], ["f1"])
        self.assertAlmostEqual(sum(cover["area_m2"] for cover in covers), 24.0, places=6)

    def test_rooms_with_nothing_above_keep_a_whole_ceiling(self):
        self.assertEqual(len(self.spec.ceiling_regions("entry")), 1)
        self.assertEqual(len(self.spec.ceiling_regions("landing")), 1)

    def test_the_fully_covered_demo_room_still_loses_its_whole_ceiling(self):
        """Regression: two identical outlines (old duplex) keep the old result."""
        demo = spec_mod.load_spec(DEMO_SPEC)
        for room_id in ("hall", "living", "kitchen", "dining"):
            with self.subTest(room=room_id):
                self.assertEqual(demo.ceiling_regions(room_id), [])
        for room_id in ("landing", "bedroom", "study", "bath"):
            with self.subTest(room=room_id):
                self.assertEqual(len(demo.ceiling_regions(room_id)), 1)

    def test_rays_land_in_the_three_regions(self):
        # covered: the f1 slab itself, one slab (0.15 m) thick
        self.assertAlmostEqual(_vertical_hit(self.caster, *self.COVERED, 1.5), 1.20, places=2)
        self.assertAlmostEqual(_vertical_hit(self.caster, *self.COVERED, 3.5, up=False), 0.65, places=2)
        # uncovered: the hall's own ceiling, 2.70 .. 2.75
        self.assertAlmostEqual(_vertical_hit(self.caster, *self.UNCOVERED, 1.5), 1.20, places=2)
        self.assertAlmostEqual(_vertical_hit(self.caster, *self.UNCOVERED, 3.5, up=False), 0.75, places=2)
        # declared void: nothing at 2.70 -- the first hit is the landing roof at 5.55
        self.assertAlmostEqual(_vertical_hit(self.caster, *self.VOID, 1.5), 4.05, places=2)

    def test_obstruction_bottoms_follows_the_same_rule(self):
        labels = [label for _bottom, label in
                  checks_mod._obstruction_bottoms(self.spec, self.COVERED, 1.5)]
        self.assertIn("slab of floor 'f1'", labels)
        self.assertNotIn("ceiling of room 'hall'", labels)
        labels = [label for _bottom, label in
                  checks_mod._obstruction_bottoms(self.spec, self.UNCOVERED, 1.5)]
        self.assertIn("ceiling of room 'hall'", labels)
        self.assertFalse(any(label.startswith("slab of floor") for label in labels))
        void = [(round(bottom, 3), label) for bottom, label in
                checks_mod._obstruction_bottoms(self.spec, self.VOID, 1.5)]
        self.assertEqual(void, [(5.55, "ceiling of room 'landing'")])

    def test_a_declared_void_stays_open(self):
        self.assertTrue(self.spec.in_slab_opening("f1", self.VOID))
        self.assertFalse(self.spec.ceiling_above("hall", self.VOID))
        self.assertFalse(self.spec.ceiling_above("hall", self.COVERED))

    def test_a_cover_in_the_middle_keeps_a_ring_of_ceiling(self):
        """The decomposition splits a ring into rectangles (no hole polygon needed)."""
        data = {
            "schema_version": 2, "unit": "meter",
            "wall_height": 2.7, "wall_thickness": 0.12,
            "floors": [{"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15},
                       {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15}],
            "rooms": [
                {"id": "low", "type": "living_room", "floor_id": "f0",
                 "polygon": [[0, 0], [6.0, 0], [6.0, 6.0], [0, 6.0]]},
                {"id": "up", "type": "bedroom", "floor_id": "f1",
                 "polygon": [[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]]},
            ],
            "cameras": [{"name": "0000", "room": "low", "pos": [0.5, 0.5, 1.5]}],
        }
        spec = spec_mod.parse_spec(data)
        parts = spec.ceiling_regions("low")
        self.assertEqual(len(parts), 4)
        self.assertAlmostEqual(sum(abs(spec_mod.polygon_area(part)) for part in parts), 32.0, places=6)
        self.assertTrue(spec.ceiling_above("low", (3.0, 5.0)))
        self.assertFalse(spec.ceiling_above("low", (3.0, 3.0)))

    def test_a_non_rectangular_outline_keeps_the_whole_ceiling(self):
        """Only axis-aligned rectangles are split; any other outline stays whole.

        The fallback is deliberate (``spec.polygon_difference``): a ceiling part
        that is not needed is a duplicated body, a missing one is a hole in the
        building, so an L-shaped room keeps its ceiling everywhere instead of
        risking a gap.
        """
        data = {
            "schema_version": 2, "unit": "meter",
            "wall_height": 2.7, "wall_thickness": 0.12,
            "floors": [{"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15},
                       {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15}],
            "rooms": [
                {"id": "low", "type": "living_room", "floor_id": "f0",
                 "polygon": [[0, 0], [4.0, 0], [4.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0, 4.0]]},
                {"id": "up", "type": "bedroom", "floor_id": "f1",
                 "polygon": [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]},
            ],
            "cameras": [{"name": "0000", "room": "low", "pos": [0.5, 3.5, 1.5]}],
        }
        spec = spec_mod.parse_spec(data)
        parts = spec.ceiling_regions("low")
        self.assertEqual(len(parts), 1)
        self.assertAlmostEqual(abs(spec_mod.polygon_area(parts[0])), 12.0, places=6)
        self.assertTrue(spec.ceiling_replaced_by("low"))
        self.assertTrue(spec.ceiling_above("low", (0.5, 3.5)))

    def test_the_minimal_review_repro_keeps_the_rest_of_the_ceiling(self):
        """Lower 4 x 4 m, upper 2 x 2 m: 12 m^2 of ceiling used to disappear."""
        data = {
            "schema_version": 2, "unit": "meter",
            "wall_height": 2.7, "wall_thickness": 0.12,
            "floors": [{"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15},
                       {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15}],
            "rooms": [
                {"id": "low", "type": "living_room", "floor_id": "f0",
                 "polygon": [[0, 0], [4.0, 0], [4.0, 4.0], [0, 4.0]]},
                {"id": "up", "type": "bedroom", "floor_id": "f1",
                 "polygon": [[0, 0], [2.0, 0], [2.0, 2.0], [0, 2.0]]},
            ],
            "cameras": [{"name": "0000", "room": "low", "pos": [3.0, 3.0, 1.5]}],
        }
        spec = spec_mod.parse_spec(data)
        parts = spec.ceiling_regions("low")
        self.assertAlmostEqual(sum(abs(spec_mod.polygon_area(part)) for part in parts), 12.0, places=6)
        self.assertTrue(spec.ceiling_above("low", (3.0, 3.0)))
        self.assertFalse(spec.ceiling_above("low", (1.0, 1.0)))
        boxes = [_box_of_polygon(part, 2.7, 2.75) for part in parts]
        boxes.append(_box(0.0, 0.0, 2.0, 2.0, 2.7, 2.85))    # the upper slab
        boxes.append(_box(0.0, 0.0, 4.0, 4.0, -0.15, 0.0))    # the lower slab
        boxes.append(_dome())
        caster = validator.MeshRayCaster(trimesh.util.concatenate(boxes))
        # the uncovered [3, 3] hits the ceiling at 2.70 (the review: "20 m, no hit")
        self.assertAlmostEqual(_vertical_hit(caster, 3.0, 3.0, 1.5), 1.20, places=2)
        # the covered [1, 1] hits the upper slab, also at 2.70
        self.assertAlmostEqual(_vertical_hit(caster, 1.0, 1.0, 1.5), 1.20, places=2)


class SettleFloorTest(unittest.TestCase):
    """Settling targets the room's own floor, not z = 0."""

    @classmethod
    def setUpClass(cls):
        cls.spec = spec_mod.parse_spec(_demo_dict())
        cls.config = settle.resolve_config(cls.spec)

    def test_an_upper_item_is_not_held_by_its_world_z(self):
        item = next(item for item in self.spec.furniture if item.name == "bed")
        placement = {"name": "bed", "z": 2.85, "mount": "floor"}
        mode, reason = settle.placement_mode(item, placement, self.config, floor_z=2.85)
        self.assertEqual(mode, "rest", reason)
        mode, reason = settle.placement_mode(item, placement, self.config, floor_z=0.0)
        self.assertEqual(mode, "held", reason)

    def test_a_wall_hung_piece_is_still_held(self):
        item = next(item for item in self.spec.furniture if item.name == "art_living")
        mode, _reason = settle.placement_mode(item, {"name": "art_living", "z": 1.62},
                                              self.config, floor_z=0.0)
        self.assertEqual(mode, "held")

    def test_plan_reads_the_elevation_from_the_spec(self):
        placements = [{"name": item.name or f"item_{index}", "asset": item.asset}
                      for index, item in enumerate(self.spec.furniture)]
        decisions = settle.plan(self.spec, placements, self.config)
        by_name = {entry["name"]: entry for entry in decisions}
        self.assertEqual(by_name["bed"]["floor_z"], 2.85)
        self.assertEqual(by_name["sofa"]["floor_z"], 0.0)
        self.assertEqual(by_name["study_desk"]["floor_z"], 2.85)
        self.assertEqual(by_name["kitchen_stove"]["floor_z"], 0.0)

    def test_ground_check_is_relative_to_the_storey_floor(self):
        config = dict(self.config)
        config["ground_tolerance_m"] = 0.02
        config["penetration_tolerance_m"] = 0.001
        on_floor = settle.ground_check(2.8501, config, floor_z=2.85)
        self.assertTrue(on_floor["grounded"])
        self.assertAlmostEqual(on_floor["bottom_m"], 0.0001, places=6)
        floating = settle.ground_check(2.90, config, floor_z=2.85)
        self.assertFalse(floating["grounded"])
        self.assertAlmostEqual(floating["floating_m"], 0.03, places=6)
        sunk = settle.ground_check(0.0, config, floor_z=2.85)
        self.assertTrue(sunk["penetrating"])


class ExportSceneTest(unittest.TestCase):
    """Per-storey plans and a viewpoint graph that spans the storeys."""

    def test_camera_adjacency_links_through_the_stair(self):
        spec = spec_mod.parse_spec(_demo_dict())
        adjacency = spec.camera_adjacency()
        self.assertIn("0100", adjacency["0000"])  # hall camera sees the landing camera's storey
        self.assertIn("0000", adjacency["0100"])
        self.assertIn("0102", adjacency["0100"])  # landing -> study door
        self.assertNotIn("0101", adjacency["0000"])  # no direct link to the bedroom

    def test_topdowns_are_written_per_storey_and_as_a_composite(self):
        spec = spec_mod.parse_spec(_demo_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "topdown.png")
            info = export_scene.draw_topdown(spec, {}, path)
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "topdown_f0.png")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "topdown_f1.png")))
            self.assertEqual([item["floor"] for item in info["floors"]], ["f0", "f1"])
            composite = export_scene.Image.open(path)
            single = export_scene.Image.open(os.path.join(tmp, "topdown_f0.png"))
            self.assertGreater(composite.size[1], 2 * single.size[1] * 0.5)

    def test_legacy_spec_writes_a_single_plan(self):
        spec = spec_mod.load_spec(os.path.join(REPO_ROOT, "configs", "blender_agent_p0_demo.json"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "topdown.png")
            info = export_scene.draw_topdown(spec, {}, path)
            self.assertTrue(os.path.isfile(path))
            self.assertNotIn("floors", info)
            self.assertEqual(os.listdir(tmp), ["topdown.png"])


class ValidatorStairsTest(unittest.TestCase):
    """The ray probes of the validator, on a synthetic two-storey shell.

    Geometry (all boxes, no booleans needed): a shared wall at x = 0 with a door
    per storey, the upper slab of floor f1 with a stairwell opening, and a
    straight 16-step stair inside it (climbing along +Y, clear of the wall).
    """

    #: stairwell outline on the upper slab (x0, y0, x1, y1)
    HOLE = (-1.5, -2.05, -0.5, 2.05)
    LOWER_DOOR_Z = 2.05
    UPPER_DOOR_Z = 2.05
    WALL_HALF = 0.06

    @classmethod
    def setUpClass(cls):
        cls.data = {
            "schema_version": 2,
            "unit": "meter", "wall_height": 2.7, "wall_thickness": 0.12,
            "floors": [{"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15},
                       {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15}],
            "rooms": [
                {"id": "a", "type": "room", "floor_id": "f0",
                 "polygon": [[-4.0, -3.0], [0.0, -3.0], [0.0, 3.0], [-4.0, 3.0]]},
                {"id": "b", "type": "room", "floor_id": "f0",
                 "polygon": [[0.0, -3.0], [4.0, -3.0], [4.0, 3.0], [0.0, 3.0]]},
                {"id": "c", "type": "room", "floor_id": "f1",
                 "polygon": [[-4.0, -3.0], [0.0, -3.0], [0.0, 3.0], [-4.0, 3.0]]},
                {"id": "d", "type": "room", "floor_id": "f1",
                 "polygon": [[0.0, -3.0], [4.0, -3.0], [4.0, 3.0], [0.0, 3.0]]},
            ],
            "connectors": [
                {"kind": "door", "from_room": "a", "to_room": "b", "from_floor": "f0",
                 "to_floor": "f0", "center": [0.0, 0.0], "width": 0.9, "height": 2.05},
                {"kind": "door", "from_room": "c", "to_room": "d", "from_floor": "f1",
                 "to_floor": "f1", "center": [0.0, 0.0], "width": 0.9, "height": 2.05},
                {"kind": "stair", "from_room": "a", "to_room": "c", "from_floor": "f0",
                 "to_floor": "f1", "width": 0.85, "run": 4.0, "rise": 2.85, "steps": 16,
                 "center": [-1.0, 0.0], "direction_deg": 90},
            ],
            "slab_openings": [{"outline": [[cls.HOLE[0], cls.HOLE[1]], [cls.HOLE[2], cls.HOLE[1]],
                                           [cls.HOLE[2], cls.HOLE[3]], [cls.HOLE[0], cls.HOLE[3]]],
                               "floors": ["f1"]}],
            "cameras": [{"name": "0000", "room": "a", "pos": [-2.0, 2.0, 1.5]},
                        {"name": "0100", "room": "c", "pos": [-2.0, 2.5, 4.35]}],
        }
        cls.spec = spec_mod.parse_spec(copy.deepcopy(cls.data))
        cls.caster = validator.MeshRayCaster(cls._mesh())

    @classmethod
    def _wall(cls, z0, z1, door_height):
        """Shared wall at x=0 (6 m long, y -3..3) with a 0.9 m door from z0."""
        half = cls.WALL_HALF
        return [
            _box(-half, -3.0, half, -0.45, z0, z1),
            _box(-half, 0.45, half, 3.0, z0, z1),
            _box(-half, -0.45, half, 0.45, z0 + door_height, z1),
        ]

    @classmethod
    def _slab_with_hole(cls):
        x0, y0, x1, y1 = cls.HOLE
        return [
            _box(-4.0, -3.0, x0, 3.0, 2.7, 2.85),
            _box(x1, -3.0, 4.0, 3.0, 2.7, 2.85),
            _box(x0, -3.0, x1, y0, 2.7, 2.85),
            _box(x0, y1, x1, 3.0, 2.7, 2.85),
        ]

    @classmethod
    def _step_boxes(cls):
        boxes = []
        for box in cls.spec.stair_plans()[0].boxes:
            xs = [point[0] for point in box["polygon"]]
            ys = [point[1] for point in box["polygon"]]
            boxes.append(_box(min(xs), min(ys), max(xs), max(ys), box["z0"], box["z1"]))
        return boxes

    @classmethod
    def _dome(cls) -> trimesh.Trimesh:
        """A far enclosure, as in a real build: the BVH keeps the room near and
        handles the dome analytically, which is the split the caster expects."""
        return trimesh.creation.icosphere(subdivisions=3, radius=30.0).apply_translation(
            (0.0, 0.0, 2.7))

    @classmethod
    def _mesh(cls):
        boxes = cls._wall(0.0, 2.7, cls.LOWER_DOOR_Z)
        boxes += cls._wall(2.85, 5.55, cls.UPPER_DOOR_Z)
        boxes.append(_box(-4.0, -3.0, 4.0, 3.0, -0.15, 0.0))  # ground floor slab
        boxes += cls._slab_with_hole()
        boxes += cls._step_boxes()
        boxes.append(cls._dome())
        return trimesh.util.concatenate(boxes)

    def test_both_storeys_doors_are_measured_in_their_own_storey(self):
        report = validator.check_door_openings(self.spec, self.caster)
        self.assertTrue(report["pass"], report["doors"])
        self.assertEqual(report["openings"], 2)
        self.assertEqual([door["floor"] for door in report["doors"]], ["f0", "f1"])
        self.assertEqual([door["floor_elevation_m"] for door in report["doors"]], [0.0, 2.85])
        for door in report["doors"]:
            self.assertAlmostEqual(door["measured_width_m"], 0.90, delta=0.05)
            self.assertAlmostEqual(door["measured_height_m"], 2.05, delta=0.06)

    def test_a_door_cut_in_the_wrong_storey_fails(self):
        # the upper storey's door was cut at the first storey's height: the hole
        # is missing from the upper wall, so the probes must reject the opening
        half = self.WALL_HALF
        boxes = self._wall(0.0, 2.7, self.LOWER_DOOR_Z)
        boxes += [_box(-half, -3.0, half, -0.45, 2.85, 5.55),
                  _box(-half, 0.45, half, 3.0, 2.85, 5.55)]
        boxes.append(_box(-4.0, -3.0, 4.0, 3.0, -0.15, 0.0))
        boxes.append(self._dome())
        report = validator.check_door_openings(self.spec, validator.MeshRayCaster(
            trimesh.util.concatenate(boxes)))
        self.assertFalse(report["pass"])
        failures = " ".join(failure for door in report["doors"] for failure in door["failures"])
        self.assertTrue("does not reach the floor" in failures or "measured opening height" in failures,
                        failures)

    def test_stair_steps_are_measured_where_the_spec_says(self):
        report = validator.check_stairs_built(self.spec, self.caster)
        self.assertTrue(report["pass"], report["stairs"][0]["failures"])
        stair = report["stairs"][0]
        self.assertEqual(len(stair["steps"]), 16)
        self.assertGreater(len(stair["opening_probes"]), 0)
        self.assertTrue(all(not probe["in_slab_band"] for probe in stair["opening_probes"]))

    def test_a_closed_slab_fails_the_opening_probe(self):
        boxes = self._wall(0.0, 2.7, self.LOWER_DOOR_Z)
        boxes += self._wall(2.85, 5.55, self.UPPER_DOOR_Z)
        boxes.append(_box(-4.0, -3.0, 4.0, 3.0, -0.15, 0.0))
        boxes.append(_box(-4.0, -3.0, 4.0, 3.0, 2.7, 2.85))  # slab without the opening
        boxes += self._step_boxes()
        boxes.append(self._dome())
        report = validator.check_stairs_built(self.spec, validator.MeshRayCaster(
            trimesh.util.concatenate(boxes)))
        self.assertFalse(report["pass"])
        self.assertTrue(any("still there" in failure for failure in report["stairs"][0]["failures"]))

    def test_the_storey_sandwich_is_one_slab_thick(self):
        report = validator.check_floor_stacks(self.spec, self.caster)
        self.assertTrue(report["pass"], report["violations"])
        pair = report["pairs"][0]
        self.assertAlmostEqual(pair["measured_ceiling_m"], 2.7, delta=0.02)
        self.assertAlmostEqual(pair["measured_surface_m"], 2.85, delta=0.02)
        self.assertAlmostEqual(pair["slab_thickness_spec_m"], 0.15, places=6)

    def test_double_slab_is_detected(self):
        boxes = self._wall(0.0, 2.7, self.LOWER_DOOR_Z)
        boxes += self._wall(2.85, 5.55, self.UPPER_DOOR_Z)
        boxes.append(_box(-4.0, -3.0, 4.0, 3.0, -0.15, 0.0))
        boxes.append(_box(-4.0, -3.0, 4.0, 3.0, 2.70, 2.85))
        boxes.append(_box(-4.0, -3.0, 4.0, 3.0, 2.45, 2.60))  # a second, lower ceiling slab
        boxes.append(self._dome())
        report = validator.check_floor_stacks(self.spec, validator.MeshRayCaster(
            trimesh.util.concatenate(boxes)))
        self.assertFalse(report["pass"])
        self.assertTrue(any("sandwich" in failure or "ceiling" in failure
                            for failure in report["violations"]))


class ContainmentTest(unittest.TestCase):
    """Furniture fits under the body that is really above it (review round 3, defect 4).

    ``ceiling_breaker.json`` is the review's repro: a 0.9 x 0.4 x 4.2 m cabinet
    against the wall of a 10 x 10 m room with a 2.70 m ceiling.  Every planar
    rule passes -- footprint, wall gap, collisions, cameras, circulation -- so
    the piece used to be accepted; the containment rule binds it to the lowest
    slab underside / owned ceiling over its own footprint and rejects it.

    ``partial_upper_floor.json`` (8 x 6 m hall on f0, the f1 landing over its
    southern half only, a 1 x 1 m declared void at 7.0/1.0) proves the rule is
    per area instead of one number per storey: the same cabinet is stopped by
    the hall's own ceiling in the uncovered half, is stopped by the f1 slab
    where the landing covers the hall, passes inside the declared void, and is
    stopped again as soon as one sample of its footprint reaches out of that
    void.  A ``mount: "ceiling"`` item has to reach the body it hangs from, so a
    pendant under the void fails instead of riding down from the landing roof.
    The final-geometry re-check (``check_final_geometry``, the CLI's
    ``--final-bboxes`` / ``--final-manifest``) re-measures the boxes as built,
    which is how a piece settle or a repair lifted into the ceiling is caught.
    """

    @classmethod
    def setUpClass(cls):
        cls.assets = checks_mod.load_assets_index(CEILING_BREAKER_ASSETS)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="containment_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    # -- helpers ----------------------------------------------------------- #
    def _entry(self, report, name="furniture_containment"):
        return next(item for item in report["checks"] if item["name"] == name)

    def _failing(self, report):
        return [item["name"] for item in report["checks"] if not item["pass"]]

    def _item(self, report, name):
        return next(item for item in self._entry(report)["items"] if item["name"] == name)

    def _breaker(self, drop_tall: bool = False):
        data = _load_dict(CEILING_BREAKER_SPEC)
        if drop_tall:
            data["furniture"] = [item for item in data["furniture"]
                                 if item["name"] != "tall_cabinet"]
        return spec_mod.parse_spec(data)

    def _partial(self, asset, pos, name, mount=None, room="hall"):
        data = _load_dict(PARTIAL_CEILING_SPEC)
        item = {"asset": asset, "room": room, "pos": list(pos), "rot_z_deg": 0, "name": name}
        if mount:
            item["mount"] = mount
        data["furniture"].append(item)
        return spec_mod.parse_spec(data)

    def _two_storeys(self, lower_asset: str):
        """Two identical 6 x 6 m rooms stacked, one item at the same XY on each."""
        data = {
            "schema_version": 2, "unit": "meter",
            "wall_height": 2.7, "wall_thickness": 0.12,
            "floors": [{"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15},
                       {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15}],
            "rooms": [
                {"id": "low", "type": "living_room", "floor_id": "f0",
                 "polygon": [[0, 0], [6.0, 0], [6.0, 6.0], [0, 6.0]]},
                {"id": "up", "type": "bedroom", "floor_id": "f1",
                 "polygon": [[0, 0], [6.0, 0], [6.0, 6.0], [0, 6.0]]},
            ],
            "cameras": [{"name": "0000", "room": "low", "pos": [3.0, 3.0, 1.5]}],
            "furniture": [
                {"asset": lower_asset, "room": "low", "pos": [4.0, 4.0], "rot_z_deg": 0,
                 "name": "low_cabinet"},
                {"asset": "ph:mid_cabinet", "room": "up", "pos": [4.0, 4.0], "rot_z_deg": 0,
                 "name": "up_cabinet"},
            ],
        }
        return spec_mod.parse_spec(data)

    # -- the review's counterexample ---------------------------------------- #
    def test_a_cabinet_taller_than_the_room_is_rejected(self):
        report = checks_mod.check_spec(self._breaker(), self.assets)
        self.assertIn("furniture_containment", self._failing(report))
        item = self._item(report, "tall_cabinet")
        self.assertAlmostEqual(item["z"][1], 4.2, places=6)
        self.assertEqual(item["lowest_face"]["kind"], "ceiling")
        self.assertAlmostEqual(item["lowest_face"]["bottom"], 2.7, places=6)
        self.assertAlmostEqual(item["clearance_m"], -1.5, places=6)
        violation = next(text for text in self._entry(report)["violations"] if "tall_cabinet" in text)
        self.assertIn("pokes 1.500 m through the ceiling of room 'living' at 2.700 m", violation)

    def test_every_other_rule_still_passes_the_same_spec(self):
        """What the review measured: only the 3D rule can see this piece."""
        report = checks_mod.check_spec(self._breaker(), self.assets)
        self.assertEqual(self._failing(report), ["furniture_containment"])

    def test_normal_furniture_in_the_same_room_passes(self):
        report = checks_mod.check_spec(self._breaker(drop_tall=True), self.assets)
        self.assertTrue(report["pass"], report["violations"])
        item = self._item(report, "low_cabinet")
        self.assertAlmostEqual(item["clearance_m"], 1.8, places=6)

    def test_the_cli_reports_the_counterexample(self):
        report_path = os.path.join(self.tmp, "checks_report.json")
        code = checks_mod.main([CEILING_BREAKER_SPEC, "--assets-index", CEILING_BREAKER_ASSETS,
                                "--report", report_path, "--quiet"])
        self.assertEqual(code, 1)
        with open(report_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertFalse(data["pass"])
        self.assertEqual([entry["name"] for entry in data["checks"] if not entry["pass"]],
                         ["furniture_containment"])

    # -- per area: the ceiling this room owns, the slab that replaces it ---- #
    def test_the_hall_ceiling_stops_a_tall_piece(self):
        spec = self._partial("ph:tall_cabinet", (7.0, 5.7), "hall_cabinet")
        report = checks_mod.check_spec(spec, self.assets)
        self.assertEqual(self._failing(report), ["furniture_containment"])
        item = self._item(report, "hall_cabinet")
        self.assertEqual(item["lowest_face"]["label"], "ceiling of room 'hall'")

    def test_where_the_landing_covers_the_hall_the_slab_binds(self):
        spec = self._partial("ph:mid_cabinet", (7.0, 2.6), "hall_cabinet")
        report = checks_mod.check_spec(spec, self.assets)
        self.assertTrue(report["pass"], report["violations"])
        item = self._item(report, "hall_cabinet")
        self.assertEqual(item["lowest_face"]["kind"], "slab")
        self.assertEqual(item["lowest_face"]["floor"], "f1")
        self.assertAlmostEqual(item["clearance_m"], 0.1, places=6)

    def test_a_declared_void_lets_the_same_piece_through(self):
        """Not one height per storey: the 4.2 m piece fits where the void is open."""
        spec = self._partial("ph:tall_cabinet", (7.0, 1.0), "hall_cabinet")
        item = self._item(checks_mod.check_spec(spec, self.assets), "hall_cabinet")
        self.assertEqual(item["lowest_face"]["label"], "ceiling of room 'landing'")
        self.assertAlmostEqual(item["lowest_face"]["bottom"], 5.55, places=6)
        self.assertAlmostEqual(item["clearance_m"], 1.35, places=6)

    def test_a_footprint_that_reaches_out_of_the_void_is_bound_again(self):
        inside = self._partial("ph:tall_cabinet", (7.0, 1.0), "hall_cabinet")
        straddling = self._partial("ph:tall_cabinet", (7.0, 1.4), "hall_cabinet")
        self.assertGreater(self._item(checks_mod.check_spec(inside, self.assets),
                                      "hall_cabinet")["clearance_m"], 1.0)
        report = checks_mod.check_spec(straddling, self.assets)
        self.assertIn("furniture_containment", self._failing(report))
        item = self._item(report, "hall_cabinet")
        self.assertEqual(item["lowest_face"]["kind"], "slab")
        self.assertAlmostEqual(item["lowest_face"]["bottom"], 2.7, places=6)

    def test_a_piece_standing_over_a_hole_in_its_own_floor_is_rejected(self):
        """Support below: the landing's own slab is opened by the declared void."""
        data = _load_dict(PARTIAL_CEILING_SPEC)
        data["furniture"].append({"asset": "ph:mid_cabinet", "room": "landing",
                                  "pos": [7.0, 1.0], "rot_z_deg": 0, "name": "void_cabinet"})
        spec = spec_mod.parse_spec(data)
        report = checks_mod.check_spec(spec, self.assets)
        self.assertIn("furniture_containment", self._failing(report))
        item = self._item(report, "void_cabinet")
        self.assertAlmostEqual(item["over_opening_share"], 1.0, places=6)
        self.assertTrue(any("over a slab opening of floor 'f1'" in text
                            for text in self._entry(report)["violations"]))

    # -- ceiling mounts hang from something real ---------------------------- #
    def test_a_pendant_lamp_reaches_the_ceiling_of_its_room(self):
        spec = self._partial("ph:ceiling_lamp", (7.0, 5.7), "hall_lamp", mount="ceiling")
        report = checks_mod.check_spec(spec, self.assets)
        self.assertTrue(report["pass"], report["violations"])
        item = self._item(report, "hall_lamp")
        self.assertAlmostEqual(item["z"][1], 2.7, places=6)
        self.assertAlmostEqual(item["clearance_m"], 0.0, places=6)
        self.assertAlmostEqual(item["own_ceiling_share"], 1.0, places=6)

    def test_a_pendant_over_the_void_has_nothing_to_hang_from(self):
        """The landing roof is metres up: the lamp cannot ride down from it."""
        spec = self._partial("ph:ceiling_lamp", (7.0, 1.0), "hall_lamp", mount="ceiling")
        report = checks_mod.check_spec(spec, self.assets)
        self.assertEqual(self._failing(report), ["furniture_containment"])
        item = self._item(report, "hall_lamp")
        self.assertAlmostEqual(item["clearance_m"], 2.85, places=6)
        violation = next(text for text in self._entry(report)["violations"] if "hall_lamp" in text)
        self.assertIn("has to reach the ceiling of its own room", violation)
        self.assertIn("declared void", violation)

    def test_a_ceiling_hung_tall_cabinet_cannot_borrow_the_void(self):
        spec = self._partial("ph:tall_cabinet", (7.0, 1.0), "hung_cabinet", mount="ceiling")
        violations = self._entry(checks_mod.check_spec(spec, self.assets))["violations"]
        self.assertTrue(any("sinks" in text for text in violations), violations)
        self.assertTrue(any("has to reach the ceiling of its own room" in text
                            for text in violations), violations)

    # -- two storeys at the same XY are measured on their own --------------- #
    def test_items_stacked_over_two_storeys_are_not_measured_against_each_other(self):
        report = checks_mod.check_spec(self._two_storeys("ph:mid_cabinet"), self.assets)
        self.assertTrue(self._entry(report)["pass"], self._entry(report)["violations"])
        self.assertTrue(self._entry(report, "furniture_no_collision")["pass"])
        low = self._item(report, "low_cabinet")
        up = self._item(report, "up_cabinet")
        self.assertEqual(low["lowest_face"]["floor"], "f1")          # bound by the f1 slab
        self.assertAlmostEqual(low["z"][1], 2.6, places=6)
        self.assertAlmostEqual(low["clearance_m"], 0.1, places=6)
        self.assertEqual(up["lowest_face"]["floor"], "f1")           # its own room's ceiling
        self.assertAlmostEqual(up["clearance_m"], 0.1, places=6)
        cross = self._entry(report, "cross_floor_xy_ok_info")
        self.assertIn(("low_cabinet", "up_cabinet"), {(pair["a"], pair["b"])
                                                     for pair in cross["pairs"]})

    def test_a_piece_that_reaches_into_the_storey_above_is_rejected(self):
        report = checks_mod.check_spec(self._two_storeys("ph:tall_cabinet"), self.assets)
        violations = self._entry(report)["violations"]
        self.assertTrue(any("low_cabinet" in text and "slab of floor 'f1'" in text
                            for text in violations), violations)
        self.assertFalse(any("up_cabinet" in text for text in violations), violations)

    # -- the final-geometry re-check ---------------------------------------- #
    def _settled_recheck(self, z_shift: float):
        """The partial scene with ``hall_cabinet`` lifted by settle / a repair."""
        spec = self._partial("ph:mid_cabinet", (7.0, 2.6), "hall_cabinet")
        authored = checks_mod.item_boxes(spec, self.assets)[0]
        self.assertTrue(checks_mod.check_spec(spec, self.assets)["pass"])
        settled = [replace(box, z=(box.z[0] + z_shift, box.z[1] + z_shift))
                   if box.name == "hall_cabinet" else box for box in authored]
        report = checks_mod.check_final_geometry(spec, settled, self.assets, authored=authored,
                                                 source="test")
        return report

    def test_the_recheck_catches_a_piece_lifted_into_the_ceiling(self):
        report = self._settled_recheck(0.4)
        self.assertEqual(report["kind"], "blender_agent_final_geometry")
        self.assertFalse(report["pass"])
        self.assertIn("furniture_containment", self._failing(report))
        adjustment = next(item for item in report["final_geometry"]["adjustments"]
                          if item["name"] == "hall_cabinet")
        self.assertTrue(adjustment["changed"])
        self.assertAlmostEqual(adjustment["z_shift_m"], 0.4, places=6)
        self.assertEqual(report["final_geometry"]["moved"], 1)

    def test_the_recheck_passes_the_settled_boxes_that_fit(self):
        report = self._settled_recheck(0.0)
        self.assertTrue(report["pass"], report["violations"])
        self.assertEqual(report["final_geometry"]["moved"], 0)

    def test_the_recheck_cli_takes_settle_boxes(self):
        spec_path = PARTIAL_CEILING_SPEC
        boxes = {"items": [
            {"name": "entry_shelf", "room": "entry", "bbox": [9.05, 0.1, 0.0, 9.95, 0.5, 1.8]},
            {"name": "hall_cabinet", "room": "hall", "bbox": [6.7, 2.35, 0.4, 7.3, 2.85, 3.0]},
        ]}
        box_path = os.path.join(self.tmp, "final_boxes.json")
        with open(box_path, "w", encoding="utf-8") as handle:
            json.dump(boxes, handle)
        report_path = os.path.join(self.tmp, "final_report.json")
        code = checks_mod.main([spec_path, "--final-bboxes", box_path,
                                "--assets-index", CEILING_BREAKER_ASSETS,
                                "--report", report_path, "--quiet"])
        self.assertEqual(code, 1)
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["kind"], "blender_agent_final_geometry")
        self.assertIn("furniture_containment", [entry["name"] for entry in report["checks"]
                                                if not entry["pass"]])
        boxes["items"][1]["bbox"] = [6.7, 2.35, 0.0, 7.3, 2.6]
        with open(box_path, "w", encoding="utf-8") as handle:
            json.dump(boxes, handle)
        self.assertEqual(checks_mod.main([spec_path, "--final-bboxes", box_path,
                                          "--assets-index", CEILING_BREAKER_ASSETS,
                                          "--quiet"]), 0)

    def test_the_recheck_reads_a_build_manifest(self):
        """The manifest path: settle's ``bbox_z``, matched to the spec by position."""
        data = _load_dict(PARTIAL_CEILING_SPEC)
        data["furniture"] += [
            {"asset": "ph:mid_cabinet", "room": "hall", "pos": [7.0, 2.6], "rot_z_deg": 0,
             "name": "cabinet"},
            {"asset": "ph:mid_cabinet", "room": "entry", "pos": [9.5, 2.4], "rot_z_deg": 180,
             "name": "cabinet"},
        ]
        manifest = {
            "kind": "blender_agent_build", "spec": data,
            "furniture": [
                {"name": "entry_shelf", "asset": "builtin:storage_shelf", "pos": [9.5, 0.3],
                 "rot_z_deg": 0, "size_m": [0.9, 0.4, 1.8], "bbox_z": [0.0, 1.8]},
                {"name": "cabinet", "asset": "ph:mid_cabinet", "pos": [7.0, 2.6], "rot_z_deg": 0,
                 "size_m": [0.6, 0.5, 2.6], "bbox_z": [0.0, 2.6]},
                {"name": "cabinet", "asset": "ph:mid_cabinet", "pos": [9.5, 2.4], "rot_z_deg": 180,
                 "size_m": [0.6, 0.5, 2.6], "bbox_z": [0.4, 3.0]},
            ],
        }
        path = os.path.join(self.tmp, "build_manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        spec, boxes, notes, source = checks_mod.load_final_boxes(path, None, self.assets)
        self.assertEqual([(box.name, box.room) for box in boxes],
                         [("entry_shelf", "entry"), ("cabinet", "hall"), ("cabinet", "entry")])
        self.assertEqual([round(box.z[0], 3) for box in boxes], [0.0, 0.0, 0.4])
        report = checks_mod.check_final_geometry(spec, boxes, self.assets, source=source)
        self.assertIn("furniture_containment", self._failing(report))
        violation = next(text for text in self._entry(report)["violations"] if "cabinet" in text)
        self.assertIn("pokes 0.300 m through the ceiling of room 'entry'", violation)

    def test_a_build_manifest_with_an_unknown_asset_is_noted_not_guessed(self):
        data = _load_dict(PARTIAL_CEILING_SPEC)
        manifest = {"kind": "blender_agent_build", "spec": data, "furniture": [
            {"name": "ghost", "asset": "ph:ghost", "pos": [9.5, 0.3], "rot_z_deg": 0,
             "size_m": [0.5, 0.5, 0.5], "bbox_z": [0.0, 0.5]}]}
        path = os.path.join(self.tmp, "build_manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        _spec, boxes, notes, _source = checks_mod.load_final_boxes(path, None, self.assets)
        self.assertEqual(boxes, [])
        self.assertTrue(any("ghost" in note for note in notes), notes)
        report = checks_mod.check_final_geometry(_spec, boxes, self.assets, notes=notes)
        self.assertIn("final_geometry_measured", self._failing(report))
        self.assertFalse(report["pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
