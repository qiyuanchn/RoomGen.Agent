"""Final-geometry operation spaces: door swing, pull-out, standing, bed side.

Run::

    python tests/test_functional_usage.py       # or: python -m unittest tests.test_functional_usage

The initial layout already reserves the clear space in front of a fixture
(``fixtures.usage_m`` / ``layout_gen.Placer.add``).  These tests freeze the
*second* half of the contract: after settle / repair / import the same space is
measured on the final boxes, and a door swing, a pulled-out chair, a bed side
or a shower entry that is now occupied fails the final-geometry check.  Four
blocked layouts are the acceptance counterexamples; a normal layout (with a
ceiling lamp and a thin rug inside the usage volume) must still pass, because
lights and floor finishes are not blockers.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import checks as checks_mod  # noqa: E402
from blender_agent import fixtures as fixtures_mod  # noqa: E402
from blender_agent import spec as spec_mod  # noqa: E402

ROOM_POLYGON = [[0.0, 0.0], [12.0, 0.0], [12.0, 10.0], [0.0, 10.0]]


def test_assets() -> dict:
    """A registry entry for a bed and a ceiling lamp (the fixtures are builtin)."""
    return {
        "test_bed": {"dims_m": (1.6, 2.0, 0.5), "categories": ["furniture", "bed"],
                     "attributes": {}, "ref": None, "source": "test"},
        "test_lamp": {"dims_m": (0.4, 0.4, 0.3), "categories": ["lighting"],
                      "attributes": {}, "ref": None, "source": "test"},
    }


def one_room_spec(furniture, camera=(1.0, 1.0, 1.5)) -> dict:
    return {
        "schema_version": 2,
        "wall_height": 2.7,
        "wall_thickness": 0.12,
        "floors": [{"id": "f1", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15}],
        "rooms": [{"id": "room", "type": "living_room", "floor_id": "f1",
                   "polygon": [list(point) for point in ROOM_POLYGON]}],
        "cameras": [{"name": "cam", "room": "room", "pos": list(camera)}],
        "furniture": list(furniture),
    }


NORMAL_FURNITURE = [
    {"asset": "builtin:wardrobe", "room": "room", "pos": [2.0, 9.62],
     "rot_z_deg": 0, "name": "wardrobe"},
    {"asset": "builtin:dining_chair", "room": "room", "pos": [5.0, 5.0],
     "rot_z_deg": 0, "name": "chair"},
    {"asset": "builtin:bath_shower", "room": "room", "pos": [0.6, 5.0],
     "rot_z_deg": 90, "name": "shower"},
    {"asset": "ph:test_bed", "room": "room", "pos": [10.0, 8.94],
     "rot_z_deg": 180, "name": "bed"},
]

WARDROBE_BLOCKED = NORMAL_FURNITURE[:1] + [
    {"asset": "builtin:storage_shelf", "room": "room", "pos": [2.0, 8.6],
     "rot_z_deg": 0, "name": "blocking_shelf"},
]

CHAIR_BLOCKED = NORMAL_FURNITURE[1:2] + [
    {"asset": "builtin:storage_shelf", "room": "room", "pos": [5.0, 5.9],
     "rot_z_deg": 180, "name": "blocking_shelf"},
]

SHOWER_BLOCKED = NORMAL_FURNITURE[2:3] + [
    {"asset": "builtin:storage_shelf", "room": "room", "pos": [1.6, 5.0],
     "rot_z_deg": 0, "name": "blocking_shelf"},
]

BED_BLOCKED = NORMAL_FURNITURE[3:4] + [
    {"asset": "builtin:wardrobe", "room": "room", "pos": [8.5, 8.5],
     "rot_z_deg": 90, "name": "block_left"},
    {"asset": "builtin:wardrobe", "room": "room", "pos": [11.5, 8.5],
     "rot_z_deg": 270, "name": "block_right"},
]


def load(data: dict):
    return spec_mod.parse_spec(data)


def final_report(spec, index=None, authored=None, boxes=None) -> dict:
    index = test_assets() if index is None else index
    boxes = checks_mod.item_boxes(spec, index)[0] if boxes is None else list(boxes)
    return checks_mod.check_final_geometry(
        spec, boxes, index, authored=boxes if authored is None else authored, source="test")


def failing(report: dict):
    return [entry["name"] for entry in report["checks"] if not entry["pass"]]


def usage_entry(report: dict) -> dict:
    return next(entry for entry in report["checks"]
                if entry["name"] == "functional_usage_clear")


class UsageDeclarationTest(unittest.TestCase):
    """The declarations are machine-readable and conservative."""

    def test_every_fixture_with_usage_declares_a_known_kind(self):
        for part, data in sorted(fixtures_mod.FIXTURE_PARTS.items()):
            declaration = data.get("usage")
            if data["usage_m"] <= 0.0:
                self.assertIsNone(declaration, part)
                continue
            self.assertIsNotNone(declaration, part)
            self.assertIn(declaration["kind"], fixtures_mod.USAGE_KINDS, part)
            self.assertAlmostEqual(declaration["depth_m"], data["usage_m"], places=9)
            self.assertIn(declaration["side"], ("front", "back", "left", "right"), part)

    def test_a_scaled_fixture_declares_a_scaled_space(self):
        declaration = fixtures_mod.usage_declaration("builtin:wardrobe", scale=2.0)
        self.assertAlmostEqual(declaration["depth_m"], 1.30, places=9)
        self.assertEqual(declaration["kind"], fixtures_mod.USAGE_DOOR_SWING)

    def test_usage_volumes_are_world_boxes_with_the_right_side(self):
        wardrobe = fixtures_mod.usage_volumes(
            "builtin:wardrobe", (1.4, 9.2925, 2.6, 9.9475), (0.0, 2.1), 0.0, 0.0)
        self.assertEqual(len(wardrobe), 1)
        self.assertEqual(wardrobe[0]["side"], "front")
        self.assertEqual(wardrobe[0]["kind"], fixtures_mod.USAGE_DOOR_SWING)
        self.assertGreater(wardrobe[0]["rect"][1], 8.6)     # in front = -Y
        self.assertLess(wardrobe[0]["rect"][3], 9.2925 + 1e-9)
        self.assertEqual(wardrobe[0]["z"], [0.0, 2.1])      # the item's own height

        chair = fixtures_mod.usage_volumes(
            "builtin:dining_chair", (4.78, 4.7725, 5.22, 5.2275), (0.0, 0.92), 0.0, 0.0)
        self.assertEqual(chair[0]["side"], "back")
        self.assertAlmostEqual(chair[0]["rect"][1], 5.2275, places=6)  # pull-out = +Y
        self.assertGreater(chair[0]["rect"][3], 5.2275)
        self.assertEqual(chair[0]["z"], [0.0, 1.1])

        bed = fixtures_mod.usage_volumes(
            "ph:GothicBed_01", (9.2, 7.94, 10.8, 9.94), (0.0, 0.5), 0.0, 0.0)
        self.assertEqual({volume["side"] for volume in bed}, {"left", "right"})
        self.assertEqual({volume["kind"] for volume in bed}, {fixtures_mod.USAGE_BED_SIDE})
        self.assertTrue(all(volume["min_clear_m"] > 0.0 for volume in bed))

    def test_a_bed_is_recognised_without_a_registry_entry(self):
        self.assertIsNotNone(fixtures_mod.usage_declaration("gen:bed_01"))
        self.assertIsNone(fixtures_mod.usage_declaration("gen:sofa_01"))


class NormalLayoutTest(unittest.TestCase):
    def test_the_normal_layout_passes_the_final_usage_check(self):
        spec = load(one_room_spec(NORMAL_FURNITURE))
        self.assertTrue(checks_mod.check_spec(spec, test_assets())["pass"])
        report = final_report(spec)
        self.assertTrue(report["pass"], report["violations"])
        entry = usage_entry(report)
        self.assertTrue(entry["pass"], entry["violations"])
        kinds = {space["kind"] for space in entry["spaces"]}
        self.assertEqual(kinds, {fixtures_mod.USAGE_DOOR_SWING, fixtures_mod.USAGE_PULL_OUT,
                                 fixtures_mod.USAGE_ENTRY, fixtures_mod.USAGE_BED_SIDE})

    def test_a_light_or_a_thin_rug_in_the_volume_does_not_block(self):
        furniture = [
            NORMAL_FURNITURE[0],
            {"asset": "builtin:rug", "room": "room", "pos": [2.0, 8.9],
             "rot_z_deg": 0, "name": "rug", "size": [0.8, 0.5]},
            {"asset": "ph:test_lamp", "room": "room", "pos": [2.0, 8.9],
             "rot_z_deg": 0, "name": "ceiling_lamp", "mount": "ceiling"},
        ]
        spec = load(one_room_spec(furniture))
        self.assertTrue(checks_mod.check_spec(spec, test_assets())["pass"])
        report = final_report(spec)
        self.assertTrue(report["pass"], report["violations"])
        wardrobe = next(space for space in usage_entry(report)["spaces"]
                        if space["item"] == "wardrobe")
        self.assertTrue(wardrobe["clear"], wardrobe["blockers"])

    def test_a_basin_beside_a_toilet_does_not_block_its_standing_spot(self):
        """Corner fixtures are the generator's normal bathroom, not a conflict."""
        furniture = [
            {"asset": "builtin:bath_toilet", "room": "room", "pos": [0.6, 0.4],
             "rot_z_deg": 90, "name": "toilet"},
            {"asset": "builtin:bath_basin", "room": "room", "pos": [1.24, 0.325],
             "rot_z_deg": 180, "name": "basin"},
        ]
        spec = load(one_room_spec(furniture, camera=(5.0, 5.0, 1.5)))
        self.assertTrue(checks_mod.check_spec(spec, test_assets())["pass"])
        report = final_report(spec)
        self.assertTrue(report["pass"], report["violations"])
        toilet = next(space for space in usage_entry(report)["spaces"]
                      if space["item"] == "toilet")
        self.assertTrue(toilet["clear"], toilet["blockers"])

    def test_a_mounted_body_without_an_inferred_placement_does_not_block(self):
        """The standalone usage path must not depend on check_spec's inference."""
        spec = load(one_room_spec([NORMAL_FURNITURE[0]]))
        wardrobe = checks_mod.item_boxes(spec, test_assets())[0][0]
        art = checks_mod.ItemBox(
            name="art", asset="ph:hanging_picture_frame_01", room="room",
            aabb=(1.6, 8.7, 2.4, 8.9), z=(1.62, 1.92), dims=None, placement="",
            rot_z_deg=0, rotation_on_grid=True)
        report = checks_mod.functional_usage_report(
            spec, [wardrobe, art], test_assets())
        self.assertTrue(report["pass"], report["violations"])
        space = next(space for space in report["spaces"] if space["item"] == "wardrobe")
        self.assertTrue(space["clear"], space["blockers"])


DESK_WITH_CHAIR = [
    {"asset": "builtin:study_desk", "room": "room", "pos": [6.0, 8.5],
     "rot_z_deg": 0, "name": "desk"},
    {"asset": "builtin:study_chair", "room": "room", "pos": [6.0, 7.89],
     "rot_z_deg": 180, "name": "desk_chair"},
]


class DeskPullOutTest(unittest.TestCase):
    """The desk's own chair is not a blocker of the desk's pull-out channel."""

    def test_a_desk_with_its_chair_passes(self):
        spec = load(one_room_spec(DESK_WITH_CHAIR))
        self.assertTrue(checks_mod.check_spec(spec, test_assets())["pass"])
        report = final_report(spec)
        self.assertTrue(report["pass"], report["violations"])
        desk = next(space for space in usage_entry(report)["spaces"]
                    if space["item"] == "desk")
        self.assertEqual(desk["kind"], fixtures_mod.USAGE_PULL_OUT)
        self.assertTrue(desk["clear"], desk["blockers"])

    def test_a_shelf_in_the_desk_channel_still_fails(self):
        furniture = [
            DESK_WITH_CHAIR[0],
            {"asset": "builtin:storage_shelf", "room": "room", "pos": [6.0, 7.9],
             "rot_z_deg": 0, "name": "shelf"},
        ]
        spec = load(one_room_spec(furniture))
        self.assertTrue(checks_mod.check_spec(spec, test_assets())["pass"])
        report = final_report(spec)
        self.assertIn("functional_usage_clear", failing(report))
        desk = next(space for space in usage_entry(report)["spaces"]
                    if space["item"] == "desk")
        self.assertTrue(any(blocker["name"] == "shelf" for blocker in desk["blockers"]))
        self.assertIn("desk", report["final_geometry"]["usage"]["blocked"])

    def test_one_clear_bed_side_is_enough_and_is_not_reported_blocked(self):
        furniture = NORMAL_FURNITURE[3:4] + [
            {"asset": "builtin:wardrobe", "room": "room", "pos": [8.5, 8.5],
             "rot_z_deg": 270, "name": "block_left"},
        ]
        spec = load(one_room_spec(furniture))
        self.assertTrue(checks_mod.check_spec(spec, test_assets())["pass"])
        report = final_report(spec)
        self.assertTrue(report["pass"], report["violations"])
        sides = {space["side"]: space for space in usage_entry(report)["spaces"]
                 if space["item"] == "bed"}
        self.assertFalse(sides["right"]["clear"])
        self.assertTrue(sides["left"]["clear"])
        self.assertNotIn("bed", report["final_geometry"]["usage"]["blocked"])


class BlockedCounterexampleTest(unittest.TestCase):
    """The four acceptance counterexamples must fail the final usage check."""

    def _assert_blocked(self, furniture, needle: str):
        spec = load(one_room_spec(furniture))
        self.assertTrue(checks_mod.check_spec(spec, test_assets())["pass"])
        report = final_report(spec)
        self.assertFalse(report["pass"], report["violations"])
        self.assertIn("functional_usage_clear", failing(report))
        entry = usage_entry(report)
        self.assertFalse(entry["pass"])
        self.assertTrue(any(needle in violation for violation in entry["violations"]),
                        entry["violations"])
        return report

    def test_a_blocked_cabinet_door_fails(self):
        report = self._assert_blocked(WARDROBE_BLOCKED, "door_swing")
        self.assertTrue(any("blocking_shelf" in violation
                            for violation in usage_entry(report)["violations"]))

    def test_a_blocked_chair_pull_out_fails(self):
        report = self._assert_blocked(CHAIR_BLOCKED, "pull_out")
        self.assertTrue(any("chair" in violation and "blocking_shelf" in violation
                            for violation in usage_entry(report)["violations"]))

    def test_a_blocked_bed_side_fails(self):
        report = self._assert_blocked(BED_BLOCKED, "no bed side")
        bed_sides = [space for space in usage_entry(report)["spaces"]
                     if space["item"] == "bed"]
        self.assertEqual(len(bed_sides), 2)
        self.assertTrue(all(not space["clear"] for space in bed_sides))
        for space in bed_sides:
            self.assertLess(space["clear_length_m"], space["required_clear_m"])

    def test_a_blocked_shower_entry_fails(self):
        report = self._assert_blocked(SHOWER_BLOCKED, "entry")
        self.assertTrue(any("shower" in violation and "blocking_shelf" in violation
                            for violation in usage_entry(report)["violations"]))

    def test_a_blocked_fridge_door_fails(self):
        report = self._assert_blocked([
            {"asset": "builtin:kitchen_fridge", "room": "room", "pos": [2.0, 9.58],
             "rot_z_deg": 0, "name": "fridge"},
            {"asset": "builtin:storage_shelf", "room": "room", "pos": [2.0, 8.8],
             "rot_z_deg": 0, "name": "blocking_shelf"},
        ], "door_swing")
        self.assertTrue(any("fridge" in violation and "blocking_shelf" in violation
                            for violation in usage_entry(report)["violations"]))


class WallAndDoorTest(unittest.TestCase):
    def test_a_usage_space_through_a_wall_fails(self):
        spec = load(one_room_spec([
            {"asset": "builtin:dining_chair", "room": "room", "pos": [2.0, 0.3],
             "rot_z_deg": 180, "name": "chair"},
        ]))
        self.assertTrue(checks_mod.check_spec(spec, test_assets())["pass"])
        report = final_report(spec)
        self.assertIn("functional_usage_clear", failing(report))
        chair = next(space for space in usage_entry(report)["spaces"]
                     if space["item"] == "chair")
        self.assertTrue(any(blocker["kind"] == "wall" for blocker in chair["blockers"]))

    def test_a_usage_space_through_a_door_opening_fails(self):
        data = {
            "schema_version": 1,
            "wall_height": 2.7,
            "wall_thickness": 0.12,
            "rooms": [
                {"id": "a", "type": "living_room",
                 "polygon": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]},
                {"id": "b", "type": "bedroom",
                 "polygon": [[0.0, -4.0], [4.0, -4.0], [4.0, 0.0], [0.0, 0.0]]},
            ],
            "openings": [{"kind": "door", "between": ["a", "b"], "center": [2.0, 0.0],
                          "width": 0.9, "height": 2.05}],
            "cameras": [{"name": "cam", "room": "a", "pos": [3.5, 3.5, 1.5]}],
            "furniture": [{"asset": "builtin:dining_chair", "room": "a",
                           "pos": [2.0, 0.45], "rot_z_deg": 180, "name": "chair"}],
        }
        spec = load(data)
        boxes = checks_mod.item_boxes(spec, {})[0]
        report = checks_mod.functional_usage_report(spec, boxes)
        chair = next(space for space in report["spaces"] if space["item"] == "chair")
        kinds = {blocker["kind"] for blocker in chair["blockers"]}
        self.assertIn("door_opening", kinds)
        self.assertNotIn("wall", kinds)


class SameNameAcrossRoomsTest(unittest.TestCase):
    def test_two_beds_named_bed_are_measured_per_room(self):
        """A clear side of one bedroom's bed must not excuse the other bed."""
        data = {
            "schema_version": 1,
            "wall_height": 2.7,
            "wall_thickness": 0.12,
            "rooms": [
                {"id": "a", "type": "bedroom",
                 "polygon": [[0.0, 0.0], [2.7, 0.0], [2.7, 4.0], [0.0, 4.0]]},
                {"id": "b", "type": "bedroom",
                 "polygon": [[2.7, 0.0], [6.7, 0.0], [6.7, 4.0], [2.7, 4.0]]},
            ],
            "cameras": [{"name": "cam", "room": "a", "pos": [1.35, 3.5, 1.5]}],
            "furniture": [
                {"asset": "ph:test_bed", "room": "a", "pos": [1.35, 2.0],
                 "rot_z_deg": 0, "name": "bed"},
                {"asset": "ph:test_bed", "room": "b", "pos": [4.7, 2.0],
                 "rot_z_deg": 0, "name": "bed"},
            ],
        }
        spec = load(data)
        boxes = checks_mod.item_boxes(spec, test_assets())[0]
        report = checks_mod.functional_usage_report(spec, boxes)
        self.assertFalse(report["pass"])
        bed_a = [space for space in report["spaces"] if space["room"] == "a"]
        self.assertEqual(len(bed_a), 2)
        self.assertTrue(all(not space["clear"] for space in bed_a), bed_a)
        bed_b = [space for space in report["spaces"] if space["room"] == "b"]
        self.assertEqual(len(bed_b), 2)
        self.assertTrue(all(space["clear"] for space in bed_b), bed_b)
        self.assertTrue(any("(a)" in violation and "no bed side" in violation
                            for violation in report["violations"]), report["violations"])
        self.assertFalse(any("(b)" in violation for violation in report["violations"]))


class AdjustmentRecordTest(unittest.TestCase):
    def test_the_final_report_keeps_the_before_after_usage_difference(self):
        spec = load(one_room_spec(WARDROBE_BLOCKED))
        boxes = checks_mod.item_boxes(spec, test_assets())[0]
        wardrobe_only = [box for box in boxes if box.name == "wardrobe"]
        report = checks_mod.check_final_geometry(
            spec, boxes, test_assets(), authored=wardrobe_only, source="test")
        usage = report["final_geometry"]["usage"]
        adjustment = next(entry for entry in usage["adjustments"]
                          if entry["item"] == "wardrobe")
        self.assertTrue(adjustment["clear_before"])
        self.assertFalse(adjustment["clear_after"])
        self.assertTrue(adjustment["changed"])
        self.assertEqual(adjustment["blockers_before"], [])
        self.assertIn("blocking_shelf", adjustment["blockers_after"])
        self.assertGreaterEqual(usage["moved"], 1)
        self.assertIn("wardrobe", usage["blocked"])


class FinalInputTest(unittest.TestCase):
    def test_a_manifest_bbox_is_used_instead_of_the_authored_footprint(self):
        spec_data = one_room_spec([NORMAL_FURNITURE[0]])
        spec = load(spec_data)
        expected = [1.4, 8.0, 0.05, 2.6, 8.65, 2.15]
        manifest = {
            "kind": "blender_agent_build",
            "spec": spec_data,
            "furniture": [{
                "name": "wardrobe", "asset": "builtin:wardrobe",
                "pos": [2.0, 9.62], "rot_z_deg": 0,
                "bbox_z": [0.0, 2.1], "bbox": expected,
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "build_manifest.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            loaded_spec, boxes, notes, _source = checks_mod.load_final_boxes(
                path, None, test_assets())
        self.assertEqual([box.name for box in boxes], ["wardrobe"])
        self.assertEqual([round(value, 4) for value in boxes[0].aabb], expected[0:2] + expected[3:5])
        self.assertEqual([round(value, 4) for value in boxes[0].z], expected[2:6:3])
        self.assertEqual(boxes[0].usage["kind"], fixtures_mod.USAGE_DOOR_SWING)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(loaded_spec.rooms[0].id, "room")
        self.assertTrue(notes == [], notes)

    def test_a_manifest_without_bbox_falls_back_to_settle_bbox_z(self):
        spec_data = one_room_spec([NORMAL_FURNITURE[0]])
        manifest = {
            "kind": "blender_agent_build",
            "spec": spec_data,
            "furniture": [{
                "name": "wardrobe", "asset": "builtin:wardrobe",
                "pos": [2.0, 9.62], "rot_z_deg": 0, "bbox_z": [0.01, 2.11],
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "build_manifest.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            _spec, boxes, notes, _source = checks_mod.load_final_boxes(
                path, None, test_assets())
        self.assertEqual([round(value, 4) for value in boxes[0].z], [0.01, 2.11])
        self.assertAlmostEqual(boxes[0].aabb[0], 1.4, places=6)
        self.assertTrue(notes == [], notes)


class CliTest(unittest.TestCase):
    def _write_index(self, directory: str) -> str:
        path = os.path.join(directory, "assets.json")
        models = {
            "test_bed": {"api_dimensions_mm": [1600.0, 2000.0, 500.0],
                         "categories": ["furniture", "bed"]},
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"models": models}, handle)
        return path

    def test_functional_usage_runs_alone_on_final_boxes(self):
        spec_path = os.path.join(TESTS_DIR, "fixtures", "two_floor_demo.json")
        with tempfile.TemporaryDirectory() as tmp:
            index_path = self._write_index(tmp)
            boxes = {"items": [
                {"name": "wardrobe", "room": "bedroom", "asset": "builtin:wardrobe",
                 "rot_z_deg": 270, "bbox": [7.29, 3.8, 2.85, 7.945, 5.0, 4.95]},
            ]}
            box_path = os.path.join(tmp, "final_boxes.json")
            with open(box_path, "w", encoding="utf-8") as handle:
                json.dump(boxes, handle)
            self.assertEqual(checks_mod.main([
                spec_path, "--final-bboxes", box_path, "--functional-usage",
                "--assets-index", index_path, "--quiet"]), 0)
            boxes["items"].append({
                "name": "blocking_shelf", "room": "bedroom",
                "asset": "builtin:storage_shelf", "rot_z_deg": 0,
                "bbox": [6.2, 4.2, 2.85, 6.75, 4.8, 4.65], "scale": 1.0,
            })
            with open(box_path, "w", encoding="utf-8") as handle:
                json.dump(boxes, handle)
            self.assertEqual(checks_mod.main([
                spec_path, "--final-bboxes", box_path, "--functional-usage",
                "--assets-index", index_path, "--quiet"]), 1)

    def test_functional_usage_runs_alone_on_a_spec(self):
        data = one_room_spec(NORMAL_FURNITURE)
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = os.path.join(tmp, "spec.json")
            index_path = self._write_index(tmp)
            with open(spec_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            self.assertEqual(checks_mod.main([
                spec_path, "--functional-usage", "--assets-index", index_path,
                "--quiet"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
