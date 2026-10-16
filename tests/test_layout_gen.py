"""Unit tests for the autonomous layout generator (host python, no bpy).

Run::

    python tests/test_layout_gen.py            # or: python tests/test_layout_gen.py -v

Covers the layout contract: deterministic seeds, room sizes and topology,
door/window geometry, the furniture placement rules that ``checks.py`` enforces
(including the ones it is supposed to *catch*), the camera stands sampled on the
walkable grid (view coverage, clearances, per-room cameras and the two-camera
bearing difference) and the pre-computed forecast of the door depth spot check
that ``tools/validate_blender_agent_p0.py`` runs after the render.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import checks as checks_mod  # noqa: E402
from blender_agent import layout_gen  # noqa: E402
from blender_agent import spec as spec_mod  # noqa: E402

ASSETS_INDEX = checks_mod.load_assets_index()
SEEDS = (0, 1, 2, 3)
HABITABLE = set(layout_gen.HABITABLE)


def _generate(seed, rooms=None):
    return layout_gen.generate(seed, rooms, ASSETS_INDEX, verbose=False)


def _check(spec_dict):
    spec = spec_mod.parse_spec(copy.deepcopy(spec_dict))
    return spec, checks_mod.check_spec(spec, ASSETS_INDEX)


def _capture(seed, rooms=None):
    """Generate a scene plus the objects the camera placement ran on.

    The camera stands are picked inside :func:`layout_gen.place_cameras`, so the
    tests re-run the placement rule against exactly the rooms, doors, items and
    shell model the generator saw (spy on the call, keep the last attempt).
    """
    captured = {}
    original = layout_gen.place_cameras

    def spy(rng, plans, doors, items, model, want_second):
        captured.update(plans=list(plans), doors=list(doors), items=list(items), model=model)
        return original(rng, plans, doors, items, model, want_second)

    layout_gen.place_cameras = spy
    try:
        scene = _generate(seed, rooms)
    finally:
        layout_gen.place_cameras = original
    return scene, captured


def _room_context(captured, room_id):
    """``(plan, items, camera context)`` for one room of a captured placement."""
    plan = next(plan for plan in captured["plans"] if plan.room_id == room_id)
    items = [item for item in captured["items"] if item.room == room_id]
    return plan, items, layout_gen._room_camera_context(plan, items, captured["doors"])


class LayoutStructureTest(unittest.TestCase):
    def test_generated_specs_parse_and_pass_checks(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                scene = _generate(seed)
                spec, report = _check(scene.spec)
                self.assertTrue(report["pass"], f"seed {seed}: {report['violations'][:3]}")
                self.assertGreaterEqual(len(spec.rooms), 2)
                self.assertLessEqual(len(spec.rooms), 5)
                types = [room.type for room in spec.rooms]
                self.assertIn("living_room", types)
                self.assertGreaterEqual(types.count("bedroom"), 1)
                self.assertEqual(len(spec.cameras), len(spec.rooms)
                                 + (1 if len(spec.rooms) <= 3 else 0))

    def test_room_areas_and_aspect_within_contract(self):
        for seed in SEEDS:
            scene = _generate(seed)
            for room in scene.spec["rooms"]:
                polygon = room["polygon"]
                width = abs(polygon[2][0] - polygon[0][0])
                depth = abs(polygon[2][1] - polygon[0][1])
                area = width * depth
                low, high = layout_gen.ROOM_AREA_M2[room["type"]]
                self.assertGreaterEqual(area, low - 1e-6, f"seed {seed} {room['id']}")
                self.assertLessEqual(area, high + 1e-6, f"seed {seed} {room['id']}")
                self.assertGreaterEqual(min(width, depth),
                                        layout_gen.ROOM_MIN_DIM_M[room["type"]] - 1e-6)

    def test_rooms_are_disjoint_and_abut(self):
        scene = _generate(0)
        rects = []
        for room in scene.spec["rooms"]:
            polygon = room["polygon"]
            rects.append((room["id"], polygon[0][0], polygon[0][1], polygon[2][0], polygon[2][1]))
        for i, first in enumerate(rects):
            for second in rects[i + 1:]:
                overlap_x = min(first[3], second[3]) - max(first[1], second[1])
                overlap_y = min(first[4], second[4]) - max(first[2], second[2])
                self.assertFalse(overlap_x > 1e-6 and overlap_y > 1e-6,
                                 f"rooms {first[0]} and {second[0]} overlap")

    def test_door_graph_is_a_tree_over_shared_walls(self):
        scene = _generate(1)
        spec = spec_mod.parse_spec(scene.spec)
        self.assertEqual(len(spec.openings), len(spec.rooms) - 1)
        adjacency = spec.room_adjacency()
        seen = {spec.rooms[0].id}
        frontier = [spec.rooms[0].id]
        while frontier:
            node = frontier.pop()
            for peer in adjacency[node]:
                if peer not in seen:
                    seen.add(peer)
                    frontier.append(peer)
        self.assertEqual(seen, {room.id for room in spec.rooms})
        segments = spec.wall_segments()
        for opening in spec.openings:
            self.assertEqual((opening.kind, opening.width, opening.height), ("door", 0.9, 2.1))
            shared = [seg for seg in segments
                      if all(room_id in seg.owners for room_id in opening.between)
                      and seg.contains_point(opening.center)]
            self.assertTrue(shared, "door centre must sit on a shared wall span")

    def test_windows_on_exterior_spans_only(self):
        for seed in SEEDS:
            scene = _generate(seed)
            spec = spec_mod.parse_spec(scene.spec)
            segments = spec.wall_segments()
            counts = {}
            for window in spec.windows:
                counts[window.room] = counts.get(window.room, 0) + 1
                self.assertAlmostEqual(window.sill, layout_gen.WINDOW_SILL, places=6)
                self.assertAlmostEqual(window.height, layout_gen.WINDOW_H, places=6)
                middle = [(window.wall[0][0] + window.wall[1][0]) / 2.0,
                          (window.wall[0][1] + window.wall[1][1]) / 2.0]
                segment = spec_mod.wall_segment_for_point(segments, middle, window.room)
                self.assertIsNotNone(segment)
                self.assertEqual(len(segment.owners), 1,
                                 "windows must sit on an exterior (single owner) wall span")
            for room_id, count in counts.items():
                room_type = spec.room(room_id).type
                self.assertLessEqual(count, 2 if room_type in HABITABLE else 1)

    def test_furniture_contract(self):
        for seed in SEEDS:
            scene = _generate(seed)
            spec = spec_mod.parse_spec(scene.spec)
            boxes, unknown = checks_mod.item_boxes(spec, ASSETS_INDEX)
            self.assertFalse(unknown)
            self.assertTrue(boxes)
            for box in boxes:
                self.assertTrue(box.rotation_on_grid, f"{box.name} rotation off the 90 degree grid")
                room = spec.room(box.room)
                gap = checks_mod.wall_gap(box.aabb, room.polygon)
                if box.placement == checks_mod.WALL_HUG:
                    self.assertLessEqual(gap, checks_mod.WALL_GAP_HUG_MAX_M + 1e-9)
                if box.placement == checks_mod.FREE_STAND:
                    self.assertGreaterEqual(gap, checks_mod.WALL_GAP_FREE_MIN_M - 1e-9)
                self.assertGreaterEqual(gap, checks_mod.WALL_GAP_MIN_M - 1e-9)
                if box.placement == checks_mod.CEILING:
                    self.assertGreaterEqual(box.z[0], checks_mod.WALK_BAND_M[1])

    def test_first_opening_is_visible_from_every_camera(self):
        for seed in SEEDS:
            scene = _generate(seed)
            report = scene.report
            self.assertEqual(report["opening_0"],
                             "|".join(scene.spec["openings"][0]["between"]))
            forecast = report["door_visual_forecast"]
            self.assertEqual(len(forecast), len(scene.spec["cameras"]))
            for camera in scene.spec["cameras"]:
                entry = forecast[camera["name"]]
                self.assertGreaterEqual(entry["ratio"], layout_gen.DOOR_FORECAST_MIN_RATIO,
                                        f"seed {seed} camera {camera['name']} sees the first opening "
                                        f"only at ratio {entry['ratio']}")

    def test_same_seed_same_scene(self):
        first = _generate(5)
        second = _generate(5)
        self.assertEqual(json.dumps(first.spec, sort_keys=True), json.dumps(second.spec, sort_keys=True))

    def test_room_count_override(self):
        for count in (2, 3, 4, 5):
            scene = _generate(11, count)
            self.assertEqual(len(scene.spec["rooms"]), count)


class ChecksCatchBadLayoutsTest(unittest.TestCase):
    """checks.py has to fail the mistakes the generator avoids."""

    def setUp(self):
        self.scene = _generate(2)
        self.spec_dict = copy.deepcopy(self.scene.spec)

    def _report(self):
        spec = spec_mod.parse_spec(copy.deepcopy(self.spec_dict))
        return checks_mod.check_spec(spec, ASSETS_INDEX)

    def _failing(self):
        try:
            report = self._report()
        except spec_mod.SpecError as error:
            return ["spec_invalid"], str(error)
        return [check["name"] for check in report["checks"] if not check["pass"]], str(report["violations"])

    def _first(self, name):
        for index, item in enumerate(self.spec_dict["furniture"]):
            if item.get("name") == name:
                return index, item
        self.fail(f"no furniture item named {name}")

    def _room_of(self, item):
        return next(room for room in self.spec_dict["rooms"] if room["id"] == item["room"])

    def test_collision_detected(self):
        _, bed = self._first("bed")
        _, plant = self._first("plant_bedroom")
        bed["pos"] = list(plant["pos"])
        failing, detail = self._failing()
        self.assertIn("furniture_no_collision", failing, detail)

    def test_item_embedded_in_wall_detected(self):
        _, sofa = self._first("sofa")
        room = self._room_of(sofa)
        sofa["pos"] = [room["polygon"][0][0] + 0.10, sofa["pos"][1]]
        failing, detail = self._failing()
        self.assertTrue({"furniture_inside_rooms", "furniture_wall_clearance"} & set(failing), detail)

    def test_camera_inside_furniture_detected(self):
        _, sofa = self._first("sofa")
        self.spec_dict["cameras"][0]["room"] = sofa["room"]
        self.spec_dict["cameras"][0]["pos"] = [sofa["pos"][0], sofa["pos"][1], 1.5]
        failing, detail = self._failing()
        self.assertIn("cameras_clear", failing, detail)

    def test_furniture_wall_across_a_room_detected(self):
        """A chain of tall furniture that cuts a room in two fails circulation."""
        room = self._room_of(self.spec_dict["furniture"][0])
        x0, y0 = room["polygon"][0]
        x1, y1 = room["polygon"][2]
        centre_x = (x0 + x1) / 2.0
        y = y0 + 0.9
        index = 0
        while y < y1 - 0.2:
            self.spec_dict["furniture"].append({
                "asset": "ph:wooden_bookshelf_worn", "room": room["id"],
                "pos": [centre_x, y], "rot_z_deg": 90, "name": f"fake_wall_{index}",
            })
            y += 1.85
            index += 1
        self.assertGreater(index, 0)
        failing, detail = self._failing()
        self.assertIn("circulation_reachable", failing, detail)

    def test_unknown_asset_detected(self):
        self.spec_dict["furniture"][0]["asset"] = "ph:not_a_real_asset"
        failing, detail = self._failing()
        self.assertIn("assets_resolved", failing, detail)

    def test_door_blocked_by_furniture_detected(self):
        opening = self.spec_dict["openings"][0]
        _, sofa = self._first("sofa")
        sofa["room"] = opening["between"][0]
        room = self._room_of(sofa)
        inner_x = (room["polygon"][0][0] + room["polygon"][2][0]) / 2.0
        inner_y = (room["polygon"][0][1] + room["polygon"][2][1]) / 2.0
        to_inside = (inner_x - opening["center"][0], inner_y - opening["center"][1])
        length = (to_inside[0] ** 2 + to_inside[1] ** 2) ** 0.5
        sofa["pos"] = [opening["center"][0] + 0.3 * to_inside[0] / length,
                       opening["center"][1] + 0.3 * to_inside[1] / length]
        sofa["rot_z_deg"] = 0
        failing, detail = self._failing()
        self.assertIn("doors_clear", failing, detail)


class CommandLineTest(unittest.TestCase):
    def test_layout_gen_cli_writes_spec_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "spec.json")
            result = subprocess.run(
                [sys.executable, os.path.join(REPO_ROOT, "blender_agent", "layout_gen.py"),
                 "--seed", "4", "--rooms", "3", "--out", out],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            with open(out, encoding="utf-8") as handle:
                spec = json.load(handle)
            self.assertEqual(len(spec["rooms"]), 3)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "layout_report.json")))
            spec_mod.parse_spec(spec)

    def test_checks_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.json")
            scene = _generate(6)
            with open(good, "w", encoding="utf-8") as handle:
                json.dump(scene.spec, handle)
            result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "blender_agent", "checks.py"),
                                     good], capture_output=True, text=True, cwd=REPO_ROOT)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:])
            self.assertIn("PASS", result.stdout)

            bad = os.path.join(tmp, "bad.json")
            broken = copy.deepcopy(scene.spec)
            broken["furniture"][0]["pos"] = [broken["rooms"][0]["polygon"][0][0],
                                             broken["rooms"][0]["polygon"][0][1]]
            with open(bad, "w", encoding="utf-8") as handle:
                json.dump(broken, handle)
            result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "blender_agent", "checks.py"),
                                     bad], capture_output=True, text=True, cwd=REPO_ROOT)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAIL", result.stdout)


class CameraSamplingTest(unittest.TestCase):
    """Cameras are sampled on the room's walkable grid and picked for what they see."""

    SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    MIN_CENTRE_DISTANCE_M = 0.40
    MIN_COVERAGE_FRACTION = 0.50

    @classmethod
    def setUpClass(cls):
        cls.scenes = {}
        cls.placed = {}
        for seed in cls.SEEDS:
            scene, captured = _capture(seed)
            cls.scenes[seed] = scene
            cls.placed[seed] = captured

    def test_every_room_has_a_camera_on_the_walkable_grid(self):
        """One camera per room, standing on a free cell the doorway BFS reaches."""
        for seed, scene in self.scenes.items():
            by_room = {}
            for camera in scene.spec["cameras"]:
                by_room.setdefault(camera["room"], []).append(camera)
            self.assertEqual(set(by_room), {room["id"] for room in scene.spec["rooms"]}, f"seed {seed}")
            for room_id, group in by_room.items():
                plan, items, context = _room_context(self.placed[seed], room_id)
                blockers = [item for item in items if layout_gen._blocks_camera(item)]
                for camera in group:
                    x, y = camera["pos"][0], camera["pos"][1]
                    box = (x, y, x, y)
                    with self.subTest(seed=seed, camera=camera["name"]):
                        self.assertTrue(layout_gen._walkable(context["grid"], x, y, reachable=True),
                                        f"seed {seed} camera {camera['name']} is not on a reachable walkable cell")
                        self.assertGreaterEqual(checks_mod.wall_gap(box, plan.rect.polygon()),
                                                layout_gen.CAMERA_INSET_M - 1e-6)
                        self.assertGreaterEqual(
                            min((checks_mod.rect_distance(box, item.aabb) for item in blockers),
                                default=float("inf")),
                            layout_gen.CAMERA_MIN_GAP_TO_FURNITURE_M - 1e-6)

    def test_cameras_leave_the_room_centre(self):
        """The replaced rule was "closest to the room centre"; no camera sits there."""
        for seed, scene in self.scenes.items():
            for camera in scene.spec["cameras"]:
                polygon = next(room["polygon"] for room in scene.spec["rooms"]
                               if room["id"] == camera["room"])
                centre = ((polygon[0][0] + polygon[2][0]) / 2.0, (polygon[0][1] + polygon[2][1]) / 2.0)
                distance = math.dist(camera["pos"][:2], centre)
                self.assertGreaterEqual(distance, self.MIN_CENTRE_DISTANCE_M,
                                        f"seed {seed} camera {camera['name']} sits on the room centre")

    def test_cameras_stay_in_the_opening_row_band(self):
        """The door check samples the mirrored column, so stands stay on the row band."""
        for seed, scene in self.scenes.items():
            door = next(door for door in self.placed[seed]["doors"]
                        if door.key() == scene.report["opening_0"])
            for camera in scene.spec["cameras"]:
                self.assertLessEqual(abs(camera["pos"][1] - door.center[1]),
                                     layout_gen.CAMERA_BAND_M + 1e-6,
                                     f"seed {seed} camera {camera['name']} left the door row band")

    def test_view_coverage_is_high_and_reported(self):
        """Every camera sees most of its room's walkable floor, and the report says so."""
        for seed, scene in self.scenes.items():
            for camera in scene.spec["cameras"]:
                plan, _items, context = _room_context(self.placed[seed], camera["room"])
                view = layout_gen._camera_view(self.placed[seed]["model"], plan, context, camera["pos"])
                entry = scene.report["door_visual_forecast"][camera["name"]]
                with self.subTest(seed=seed, camera=camera["name"]):
                    self.assertAlmostEqual(entry["coverage"]["visible_area_fraction"],
                                           round(view["coverage"], 3), places=3)
                    self.assertAlmostEqual(entry["score"], round(view["score"], 3), places=3)
                    self.assertGreaterEqual(view["coverage"], self.MIN_COVERAGE_FRACTION)

    def test_two_camera_rooms_look_from_different_sides(self):
        """A two-camera room gets two stands with a real bearing difference."""
        for seed, scene in self.scenes.items():
            rooms = scene.spec["rooms"]
            cameras = scene.spec["cameras"]
            self.assertEqual(len(cameras), len(rooms) + (1 if len(rooms) <= 3 else 0), f"seed {seed}")
            for room in rooms:
                group = [camera for camera in cameras if camera["room"] == room["id"]]
                if len(group) < 2:
                    continue
                first, second = group
                with self.subTest(seed=seed, room=room["id"]):
                    self.assertGreaterEqual(math.dist(first["pos"][:2], second["pos"][:2]),
                                            layout_gen.CAMERA_SECOND_MIN_GAP_M - 1e-6)
                    entry = scene.report["door_visual_forecast"][second["name"]]["second_camera"]
                    self.assertGreaterEqual(
                        entry["bearing_angle_deg"], layout_gen.CAMERA_SECOND_MIN_ANGLE_DEG,
                        f"seed {seed} room {room['id']}: the second camera looks at the room from "
                        f"only {entry['bearing_angle_deg']} deg away")


class ShellModelTest(unittest.TestCase):
    def test_forecast_matches_the_documented_validator_formula(self):
        """The model reproduces the validator's column and depth arithmetic."""
        scene = _generate(0)
        spec = spec_mod.parse_spec(scene.spec)
        rooms = {room.id: layout_gen.Rect(room.polygon[0][0], room.polygon[0][1],
                                          room.polygon[2][0], room.polygon[2][1]) for room in spec.rooms}
        doors = []
        for opening in spec.openings:
            segments = spec.wall_segments()
            segment = spec_mod.wall_segment_for_point(segments, opening.center, opening.between[0])
            wall = layout_gen.Wall("N" if abs(segment.direction[1]) > 0.5 else "E", 0.0, 0.0, 0.0)
            doors.append(layout_gen.DoorPlan(a=opening.between[0], b=opening.between[1],
                                             center=tuple(opening.center),
                                             span=(opening.center[0] - 0.45, opening.center[0] + 0.45),
                                             wall=wall, direction=segment.direction))
        model = layout_gen.ShellModel(rooms, doors, [])
        camera = spec.cameras[0]
        forecast = model.ray_for_pixel(doors[0], tuple(camera.pos))
        self.assertIsNotNone(forecast)
        # the ray through the doorway has to travel further than the wall above it
        self.assertGreater(forecast["door_depth_m"], forecast["wall_depth_m"])
        self.assertEqual(forecast["column"], forecast["column"] % spec_mod.PANO_WIDTH)

    def test_asset_index_reports_metres(self):
        sofa = ASSETS_INDEX["sofa_02"]["dims_m"]
        self.assertAlmostEqual(sofa[0], 1.8072, places=4)
        self.assertGreater(sofa[0], 1.0)
        self.assertLess(sofa[0], 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
