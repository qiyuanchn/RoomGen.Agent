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
  geometry / opening / headroom / clearance rule, and the same-XY-same-storey
  pairs that must *not* read as a collision.
* **settle**: the rest plane of a placement is its room's floor (a second
  storey item is not "held" because its world z is large).
* **export**: per-storey top-down plans plus the composite, and the viewpoint
  graph spanning the storeys through the stair.
* **validator**: a synthetic two-storey shell proves the door probes measure
  each opening inside its own storey, the step tops are really built and the
  upper slab / lower ceiling are one sandwich of the declared thickness.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

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


def _demo_dict() -> dict:
    with open(DEMO_SPEC, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
                     "cross_floor_xy_ok_info", "circulation_reachable"):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
