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

    def test_door_visibility_is_reported_per_camera_not_enforced(self):
        """Every camera reports which of *its* doors it sees; nothing is gated.

        The old contract was "``spec.openings[0]`` is visible from every camera
        at ratio >= 1.65", which the validator measured on a mirrored column and
        which pinned all doors and cameras to one Y line.  The scene gate is now
        the per-opening geometry check; visibility is coverage.
        """
        for seed in SEEDS:
            scene = _generate(seed)
            report = scene.report
            forecast = report["door_visual_forecast"]
            self.assertEqual(len(forecast), len(scene.spec["cameras"]))
            rooms = {room["id"] for room in scene.spec["rooms"]}
            for camera in scene.spec["cameras"]:
                entry = forecast[camera["name"]]
                self.assertEqual(entry["room"], camera["room"])
                self.assertIn(entry["room"], rooms)
                self.assertLessEqual(entry["doors_visible"], entry["doors_checked"])
                self.assertEqual(len(entry["doors"]), entry["doors_checked"])
                for door in entry["doors"]:
                    self.assertIn(camera["room"], door["between"])
                    self.assertIsInstance(door["visible"], bool)
                    self.assertIsNotNone(door["ratio"])
            opening0 = report["opening_0"]
            if opening0 is not None:
                keys = {"|".join(opening["between"]) for opening in scene.spec["openings"]}
                self.assertIn(opening0, keys)

    def test_every_camera_is_the_best_scoring_stand_of_its_room(self):
        """The stand sampling scores the whole walkable area, not a band.

        Only the first camera of a room is the plain argmax; a second camera is
        picked by the two-camera rule (gap + bearing), which is checked separately.
        """
        for seed in SEEDS:
            scene, captured = _capture(seed)
            first_of_room = {}
            for camera in scene.spec["cameras"]:
                first_of_room.setdefault(camera["room"], camera)
            for camera in scene.spec["cameras"]:
                if first_of_room[camera["room"]] is not camera:
                    continue
                plan, items, context = _room_context(captured, camera["room"])
                doors = [door for door in captured["doors"] if camera["room"] in (door.a, door.b)]
                stands = layout_gen._camera_stands(plan, items, context["grid"])

                def score(pos):
                    view = layout_gen._camera_view(captured["model"], plan, context, pos)
                    seen = sum(1 for door in doors if layout_gen.door_sight_line(
                        captured["model"], door, (pos[0], pos[1], layout_gen.CAMERA_Z_M)))
                    return view["score"] + layout_gen.CAMERA_DOOR_SIGHT_WEIGHT * (
                        seen / len(doors) if doors else 0.0)

                with self.subTest(seed=seed, camera=camera["name"]):
                    self.assertGreater(len(stands), 1)
                    # candidate stands cover the room in both axes: no door-line band
                    self.assertGreater(max(y for _x, y in stands) - min(y for _x, y in stands), 0.25)
                    picked = (camera["pos"][0], camera["pos"][1])
                    self.assertIn(picked, stands)
                    self.assertGreaterEqual(score(picked) + 1e-9, max(score(pos) for pos in stands),
                                            f"seed {seed} camera {camera['name']} is not the best stand")

    def test_same_seed_same_scene(self):
        first = _generate(5)
        second = _generate(5)
        self.assertEqual(json.dumps(first.spec, sort_keys=True), json.dumps(second.spec, sort_keys=True))

    def test_room_count_override(self):
        for count in (2, 3, 4, 5):
            scene = _generate(11, count)
            self.assertEqual(len(scene.spec["rooms"]), count)

    def test_door_position_is_sampled_along_the_whole_shared_wall(self):
        """No common door line: the door may sit anywhere on its shared wall.

        The old rule clamped every door to ``y_star +- 0.12`` around one line for
        the whole plan; the position is now sampled over the full span (minus a
        jamb's margin), which is what an rng that always picks the top of the
        range shows.
        """
        plans = layout_gen.plan_rooms(layout_gen.random.Random(0),
                                      ["living_room", "bedroom", "kitchen", "study"])

        class _TopEnd:
            def uniform(self, low, high):
                return high

        doors = layout_gen.plan_doors(_TopEnd(), plans)
        self.assertGreaterEqual(len(doors), 2)
        margin = layout_gen.DOOR_W / 2.0 + 0.20
        for door in doors:
            along = door.center[0] if door.wall.side in ("N", "S") else door.center[1]
            with self.subTest(between=(door.a, door.b)):
                self.assertAlmostEqual(along, round(door.wall.hi - margin, 2), places=2)


class BlockShapeTest(unittest.TestCase):
    """``--shape block``: the same room chain, wrapped into two dimensions."""

    SEEDS = (0, 1, 2, 4, 8)

    @classmethod
    def setUpClass(cls):
        cls.scenes = {seed: layout_gen.generate(seed, None, ASSETS_INDEX, verbose=False, shape="block")
                      for seed in cls.SEEDS}

    def test_block_layouts_are_two_dimensional_and_pass_checks(self):
        for seed, scene in self.scenes.items():
            spec = spec_mod.parse_spec(copy.deepcopy(scene.spec))
            report = checks_mod.check_spec(spec, ASSETS_INDEX)
            rows = sorted({round(room["rect"][1], 2) for room in scene.report["rooms"]})
            with self.subTest(seed=seed):
                self.assertTrue(report["pass"], report["violations"][:3])
                self.assertGreaterEqual(len(rows), 2, "the block is still one row")
                self.assertEqual(len(spec.cameras), len(spec.rooms)
                                 + (1 if len(spec.rooms) <= 3 else 0))
                self.assertEqual(scene.report["shape"], "block")
                self.assertEqual(scene.report["stats"]["shape"], "block")

    def test_block_room_graph_is_still_a_path_over_shared_walls(self):
        for seed, scene in self.scenes.items():
            spec = spec_mod.parse_spec(copy.deepcopy(scene.spec))
            adjacency = spec.room_adjacency()
            with self.subTest(seed=seed):
                self.assertEqual(len(spec.openings), len(spec.rooms) - 1)
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
                    shared = [seg for seg in segments
                              if all(room_id in seg.owners for room_id in opening.between)
                              and seg.contains_point(opening.center)]
                    self.assertTrue(shared, f"door {opening.between} must sit on a shared wall span")

    def test_rooms_keep_their_area_and_do_not_overlap(self):
        for seed, scene in self.scenes.items():
            rects = []
            for room in scene.spec["rooms"]:
                polygon = room["polygon"]
                low, high = layout_gen.ROOM_AREA_M2[room["type"]]
                area = abs((polygon[2][0] - polygon[0][0]) * (polygon[2][1] - polygon[0][1]))
                with self.subTest(seed=seed, room=room["id"]):
                    self.assertGreaterEqual(area, low - 1e-6)
                    self.assertLessEqual(area, high + 1e-6)
                rects.append((room["id"], polygon[0][0], polygon[0][1], polygon[2][0], polygon[2][1]))
            for index, first in enumerate(rects):
                for second in rects[index + 1:]:
                    overlap_x = min(first[3], second[3]) - max(first[1], second[1])
                    overlap_y = min(first[4], second[4]) - max(first[2], second[2])
                    with self.subTest(seed=seed, rooms=(first[0], second[0])):
                        self.assertFalse(overlap_x > 1e-6 and overlap_y > 1e-6,
                                         f"rooms {first[0]} and {second[0]} overlap")

    def test_block_and_row_shapes_differ(self):
        row = layout_gen.generate(0, None, ASSETS_INDEX, verbose=False, shape="row")
        block = self.scenes[0]
        self.assertNotEqual(json.dumps(row.spec, sort_keys=True), json.dumps(block.spec, sort_keys=True))
        self.assertEqual(len({round(room["rect"][1], 2) for room in row.report["rooms"]}), 1)
        self.assertEqual(row.report["shape"], "row")


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

    def _any_blocking(self, room_id, exclude=()):
        """Any blocking item of a room, whatever the slots happened to place."""
        for index, item in enumerate(self.spec_dict["furniture"]):
            if item["room"] != room_id or item.get("name") in exclude:
                continue
            if item.get("mount") in ("ceiling", "wall") or item.get("size") is not None:
                continue
            if item.get("asset") == "builtin:rug":
                continue
            return index, item
        self.fail(f"no blocking furniture in room {room_id}")

    def _room_of(self, item):
        return next(room for room in self.spec_dict["rooms"] if room["id"] == item["room"])

    def test_collision_detected(self):
        _index, bed = self._first("bed")
        _other_index, other = self._any_blocking(bed["room"], exclude={bed.get("name")})
        bed["pos"] = list(other["pos"])
        bed["rot_z_deg"] = other.get("rot_z_deg", 0.0)
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
            item = broken["furniture"][0]
            room = next(room for room in broken["rooms"] if room["id"] == item["room"])
            item["pos"] = [room["polygon"][0][0] + 0.02, room["polygon"][0][1] + 0.02]
            with open(bad, "w", encoding="utf-8") as handle:
                json.dump(broken, handle)
            result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "blender_agent", "checks.py"),
                                     bad], capture_output=True, text=True, cwd=REPO_ROOT)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAIL", result.stdout)


class CameraSamplingTest(unittest.TestCase):
    """Cameras are sampled on the room's walkable grid and picked for what they see."""

    SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
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

    def test_view_coverage_is_high_and_reported(self):
        """Every camera sees most of its room's walkable floor, and the report says so.

        "Most" is what the room can offer: a small, fully furnished bathroom has
        furniture (and the doorway reveal) hiding a good part of its floor from
        *every* stand, so where no stand reaches ``MIN_COVERAGE_FRACTION`` the
        camera has to be within 5 % of the best stand the room has.
        """
        for seed, scene in self.scenes.items():
            for camera in scene.spec["cameras"]:
                plan, items, context = _room_context(self.placed[seed], camera["room"])
                view = layout_gen._camera_view(self.placed[seed]["model"], plan, context, camera["pos"])
                entry = scene.report["door_visual_forecast"][camera["name"]]
                with self.subTest(seed=seed, camera=camera["name"]):
                    self.assertAlmostEqual(entry["coverage"]["visible_area_fraction"],
                                           round(view["coverage"], 3), places=3)
                    self.assertAlmostEqual(entry["score"], round(view["score"], 3), places=3)
                    if view["coverage"] >= self.MIN_COVERAGE_FRACTION:
                        continue
                    best = max(layout_gen._camera_view(
                        self.placed[seed]["model"], plan, context, stand)["coverage"]
                        for stand in layout_gen._camera_stands(plan, items, context["grid"]))
                    self.assertGreaterEqual(
                        view["coverage"], 0.95 * best,
                        f"seed {seed} camera {camera['name']} sees {view['coverage']:.3f} of "
                        f"{plan.room_id}, while a stand in the room reaches {best:.3f}")

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
    """The forecast follows the validator's projection, not the mirrored column."""

    def _model_and_doors(self, seed=0, shape="row"):
        scene = layout_gen.generate(seed, None, ASSETS_INDEX, verbose=False, shape=shape)
        spec = spec_mod.parse_spec(scene.spec)
        rooms = {room.id: layout_gen.Rect(room.polygon[0][0], room.polygon[0][1],
                                          room.polygon[2][0], room.polygon[2][1]) for room in spec.rooms}
        doors = []
        for opening in spec.openings:
            segments = spec.wall_segments()
            segment = spec_mod.wall_segment_for_point(segments, opening.center, opening.between[0])
            side = "N" if abs(segment.direction[1]) > 0.5 else "E"
            along = opening.center[0] if side in ("N", "S") else opening.center[1]
            line = opening.center[1] if side in ("N", "S") else opening.center[0]
            doors.append(layout_gen.DoorPlan(a=opening.between[0], b=opening.between[1],
                                             center=tuple(opening.center),
                                             span=(along - 0.45, along + 0.45),
                                             wall=layout_gen.Wall(side, line, 0.0, 0.0),
                                             direction=segment.direction))
        return spec, layout_gen.ShellModel(rooms, doors, []), doors

    def test_forecast_projects_the_door_with_the_camera_matrix(self):
        for shape in ("row", "block"):
            spec, model, doors = self._model_and_doors(seed=0, shape=shape)
            for camera in spec.cameras:
                for door in doors:
                    if camera.room not in (door.a, door.b):
                        continue
                    forecast = model.ray_for_pixel(door, tuple(camera.pos))
                    with self.subTest(shape=shape, camera=camera.name, door=door.key()):
                        self.assertIsNotNone(forecast)
                        point = (door.center[0], door.center[1], layout_gen.DOOR_H / 2.0)
                        column, _row = layout_gen.erp_pixel_of_direction(
                            tuple(point[i] - camera.pos[i] for i in range(3)))
                        self.assertEqual(forecast["column"], column % spec_mod.PANO_WIDTH)
                        self.assertGreater(forecast["depth_m"], forecast["expected_m"] - 1e-6)

    def test_door_sight_is_visible_from_the_middle_of_the_room(self):
        """A camera in the open part of a room sees its doorway, and the model says so."""
        for shape in ("row", "block"):
            spec, model, doors = self._model_and_doors(seed=0, shape=shape)
            by_room = {}
            for door in doors:
                by_room.setdefault(door.a, []).append(door)
                by_room.setdefault(door.b, []).append(door)
            for room in spec.rooms:
                if room.id not in by_room:
                    continue
                plan_rect = layout_gen.Rect(room.polygon[0][0], room.polygon[0][1],
                                            room.polygon[2][0], room.polygon[2][1])
                # stand on a walkable cell of the room and require one door to be visible
                grid = layout_gen._walkable_grid(
                    layout_gen.RoomPlan(room_id=room.id, room_type=room.type, rect=plan_rect,
                                        target_area=plan_rect.area, width=plan_rect.w, depth=plan_rect.h),
                    [])
                layout_gen._mark_reachable(grid, [door.center for door in by_room[room.id]])
                stands = layout_gen._camera_stands(
                    layout_gen.RoomPlan(room_id=room.id, room_type=room.type, rect=plan_rect,
                                        target_area=plan_rect.area, width=plan_rect.w, depth=plan_rect.h),
                    [], grid)
                self.assertTrue(stands, f"{shape}: no stand in {room.id}")
                seen = [pos for pos in stands
                        if any(layout_gen.door_sight_line(model, door, (pos[0], pos[1], layout_gen.CAMERA_Z_M))
                               for door in by_room[room.id])]
                with self.subTest(shape=shape, room=room.id):
                    self.assertTrue(seen, f"{shape}: no stand in {room.id} sees one of its doorways")

    def test_asset_index_reports_metres(self):
        sofa = ASSETS_INDEX["sofa_02"]["dims_m"]
        self.assertAlmostEqual(sofa[0], 1.8072, places=4)
        self.assertGreater(sofa[0], 1.0)
        self.assertLess(sofa[0], 3.0)


class TopologyFamilyTest(unittest.TestCase):
    """The topology families of layout_gen.ROOM_FAMILIES.

    Every family is a different arrangement of the same functional programme
    (living room, bedroom, kitchen, bathroom, ...) on a rectangle grid: a 2x2
    block, an L, a central living room with rooms around it, one- and two-sided
    corridors and a circulation loop.  A layout is only accepted when its door
    graph satisfies :func:`layout_gen.door_semantics` and ``checks.py`` passes.
    """

    FAMILIES = ("compact_2x2", "l_shape", "central_living",
                "corridor_single", "corridor_double", "loop")
    SEEDS = (0, 1, 2)

    @classmethod
    def setUpClass(cls):
        cls.scenes = {}
        for family in cls.FAMILIES:
            for seed in cls.SEEDS:
                cls.scenes[(family, seed)] = layout_gen.generate(
                    seed, None, ASSETS_INDEX, attempts=80, verbose=False, shape=family)

    def test_every_family_and_seed_passes_checks(self):
        for (family, seed), scene in self.scenes.items():
            report = scene.report["checks"]
            with self.subTest(family=family, seed=seed):
                self.assertTrue(report["pass"], report["violations"][:4])

    def test_semantic_rules_hold(self):
        """Kitchen next to dining, bathroom private, bedroom not a thoroughfare."""
        for (family, seed), scene in self.scenes.items():
            semantics = scene.report["door_semantics"]
            types = {room["id"]: room["type"] for room in scene.spec["rooms"]}
            with self.subTest(family=family, seed=seed):
                self.assertTrue(semantics["pass"], semantics["violations"])
                self.assertTrue(semantics["bathrooms_are_leaves"])
                self.assertTrue(semantics["public_rooms_connected"])
                for room_id, degree in semantics["degrees"].items():
                    if types[room_id] == "bathroom":
                        self.assertEqual(degree, 1, f"bathroom {room_id} has {degree} doors")
                if "dining" in types.values():
                    self.assertIn(semantics["kitchen_next_to"], ("dining", "living_room"))

    def test_forbidden_door_pairs_never_appear(self):
        """No bedroom is entered from a kitchen; two private rooms are not linked."""
        forbidden = {("kitchen", "bedroom"), ("kitchen", "bathroom"),
                     ("bedroom", "bedroom"), ("bathroom", "bathroom")}
        for (family, seed), scene in self.scenes.items():
            types = {room["id"]: room["type"] for room in scene.spec["rooms"]}
            for opening in scene.spec["openings"]:
                pair = tuple(sorted(types[room_id] for room_id in opening["between"]))
                with self.subTest(family=family, seed=seed, between=opening["between"]):
                    self.assertNotIn(pair, forbidden)

    def test_programme_covers_living_kitchen_bedroom_bathroom(self):
        for (family, seed), scene in self.scenes.items():
            types = [room["type"] for room in scene.spec["rooms"]]
            with self.subTest(family=family, seed=seed):
                for room_type in ("living_room", "kitchen", "bedroom", "bathroom"):
                    self.assertIn(room_type, types)

    def test_room_rectangles_respect_the_bands(self):
        for (family, seed), scene in self.scenes.items():
            for room in scene.spec["rooms"]:
                polygon = room["polygon"]
                width = abs(polygon[2][0] - polygon[0][0])
                depth = abs(polygon[2][1] - polygon[0][1])
                low, high = layout_gen.ROOM_AREA_M2[room["type"]]
                with self.subTest(family=family, seed=seed, room=room["id"]):
                    self.assertGreaterEqual(width * depth, low - 1e-6)
                    self.assertLessEqual(width * depth, high + 1e-6)
                    self.assertGreaterEqual(min(width, depth),
                                            layout_gen.ROOM_MIN_DIM_M[room["type"]] - 1e-6)

    def test_rooms_are_disjoint_rectangles(self):
        for (family, seed), scene in self.scenes.items():
            rects = []
            for room in scene.spec["rooms"]:
                xs = [point[0] for point in room["polygon"]]
                ys = [point[1] for point in room["polygon"]]
                rects.append((room["id"], min(xs), min(ys), max(xs), max(ys)))
            for index, first in enumerate(rects):
                for second in rects[index + 1:]:
                    overlap_x = min(first[3], second[3]) - max(first[1], second[1])
                    overlap_y = min(first[4], second[4]) - max(first[2], second[2])
                    self.assertFalse(overlap_x > 1e-6 and overlap_y > 1e-6,
                                     f"{family} {seed}: {first[0]} overlaps {second[0]}")

    def test_no_phantom_wall_spans(self):
        """Every wall span is owned by at least one room.

        ``spec.build_wall_segments`` used to emit a segment for the gap between
        two covered spans of the same line, which drew a wall across the middle
        of a room in the 2x2 / central-living families.
        """
        for (family, seed), scene in self.scenes.items():
            spec = spec_mod.parse_spec(copy.deepcopy(scene.spec))
            for segment in spec.wall_segments():
                self.assertTrue(segment.owners,
                                f"{family} {seed}: ownerless wall span {segment.p0} .. {segment.p1}")

    def test_loop_family_has_a_cycle(self):
        scene = self.scenes[("loop", 1)]
        spec = spec_mod.parse_spec(scene.spec)
        # a ring: n doors over n rooms, so the door graph has one cycle
        self.assertGreaterEqual(len(spec.openings), len(spec.rooms))
        adjacency = spec.room_adjacency()
        # the living room and the kitchen are connected twice (through the ring)
        ways = 0
        start = spec.room("living").id
        target = spec.room("kitchen").id
        frontier = [(start, [start])]
        while frontier:
            room_id, path = frontier.pop()
            if len(path) > 4:
                continue
            for peer in adjacency[room_id]:
                if peer in path:
                    continue
                if peer == target:
                    ways += 1
                    continue
                frontier.append((peer, path + [peer]))
        self.assertGreaterEqual(ways, 1)

    def test_families_are_deterministic(self):
        for family in self.FAMILIES:
            first = layout_gen.generate(4, None, ASSETS_INDEX, verbose=False, shape=family)
            second = layout_gen.generate(4, None, ASSETS_INDEX, verbose=False, shape=family)
            self.assertEqual(json.dumps(first.spec, sort_keys=True),
                             json.dumps(second.spec, sort_keys=True), family)

    def test_auto_shape_draws_families_from_the_seed(self):
        shapes = {}
        for seed in range(10):
            scene = layout_gen.generate(seed, None, ASSETS_INDEX, attempts=80, verbose=False, shape="auto")
            shapes[scene.report["shape"]] = shapes.get(scene.report["shape"], 0) + 1
        self.assertGreaterEqual(len(shapes), 3, shapes)

    def test_forced_family_reports_its_name(self):
        scene = layout_gen.generate(0, None, ASSETS_INDEX, verbose=False, shape="corridor_double")
        self.assertEqual(scene.report["shape"], "corridor_double")
        self.assertEqual(scene.report["family"], "corridor_double")
        self.assertEqual(len(scene.spec["rooms"]), layout_gen.ROOM_FAMILIES["corridor_double"].count)

    def test_functional_rooms_are_reachable(self):
        """Entry, storage, balcony, laundry and dining appear as the flat grows."""
        seen: dict = {}
        for seed in (0, 1, 8, 10, 21):
            scene = layout_gen.generate(seed, 5, ASSETS_INDEX, attempts=80, verbose=False, shape="row")
            for room in scene.spec["rooms"]:
                seen[room["type"]] = seen.get(room["type"], 0) + 1
        for room_type in ("entry", "storage", "balcony", "laundry", "dining"):
            with self.subTest(room=room_type):
                self.assertGreaterEqual(seen.get(room_type, 0), 1, seen)

    def test_windowless_service_rooms_get_a_lamp(self):
        """A bathroom, hall or storeroom is lit by its doorway alone otherwise."""
        for (family, seed), scene in self.scenes.items():
            types = {room["id"]: room["type"] for room in scene.spec["rooms"]}
            lamps = {item["room"] for item in scene.spec["furniture"]
                     if item.get("mount") == "ceiling" or "lamp" in item["asset"]}
            dark = [room_id for room_id, room_type in types.items()
                    if room_type in ("bathroom", "corridor", "entry", "storage", "laundry")
                    and room_id not in lamps]
            with self.subTest(family=family, seed=seed):
                self.assertEqual(dark, [], f"{family} {seed}: unlit rooms {dark}")

    def test_room_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            layout_gen.generate(0, 3, ASSETS_INDEX, verbose=False, shape="compact_2x2")


class FixtureFurnitureTest(unittest.TestCase):
    """The ``builtin:`` functional furniture of blender_agent/fixtures.py."""

    def test_registry_dims_are_the_box_union(self):
        from blender_agent import fixtures
        for name, part in fixtures.FIXTURE_PARTS.items():
            boxes = part["boxes"]
            xs = [box[2][0] + sign * box[3][0] / 2.0 for box in boxes for sign in (-1, 1)]
            ys = [box[2][1] + sign * box[3][1] / 2.0 for box in boxes for sign in (-1, 1)]
            z_bottom = min(box[2][2] for box in boxes)
            z_top = max(box[2][2] + box[3][2] for box in boxes)
            union = (round(max(xs) - min(xs), 4), round(max(ys) - min(ys), 4), round(z_top - z_bottom, 4))
            with self.subTest(part=name):
                self.assertEqual(union, part["dims"])
                self.assertAlmostEqual(min(xs) + max(xs), 0.0, places=6)
                self.assertAlmostEqual(min(ys) + max(ys), 0.0, places=6)
                self.assertAlmostEqual(z_bottom, 0.0, places=6)

    def test_checks_sizes_every_fixture_part(self):
        from blender_agent import fixtures
        for asset in fixtures.BUILTIN_DIMS_M:
            with self.subTest(asset=asset):
                self.assertIsNotNone(checks_mod.item_dims(asset, ASSETS_INDEX))

    def test_functional_rooms_get_their_fixtures(self):
        scene = layout_gen.generate(0, None, ASSETS_INDEX, verbose=False, shape="central_living")
        types = {room["id"]: room["type"] for room in scene.spec["rooms"]}
        assets = [item["asset"] for item in scene.spec["furniture"]]
        for asset in ("builtin:kitchen_cabinet", "builtin:kitchen_stove", "builtin:kitchen_sink",
                      "builtin:bath_toilet", "builtin:bath_basin", "builtin:bath_shower"):
            self.assertIn(asset, assets)
        for item in scene.spec["furniture"]:
            if item["asset"].startswith("builtin:kitchen") and item["asset"] != "builtin:kitchen_upper":
                self.assertEqual(types[item["room"]], "kitchen")
            if item["asset"].startswith("builtin:bath_"):
                self.assertEqual(types[item["room"]], "bathroom")

    def test_study_has_a_desk_and_a_bookshelf(self):
        scene = layout_gen.generate(0, None, ASSETS_INDEX, verbose=False, shape="l_shape")
        spec = spec_mod.parse_spec(copy.deepcopy(scene.spec))
        by_room = {}
        for item in spec.furniture:
            by_room.setdefault(spec.room(item.room).type, []).append(item.asset)
        if "study" in by_room:
            self.assertIn("builtin:study_desk", by_room["study"])
            self.assertTrue(any("bookshelf" in asset for asset in by_room["study"]),
                            f"no bookshelf in the study: {by_room['study']}")
        if "dining" in by_room:
            self.assertIn("builtin:dining_table", by_room["dining"])
            self.assertGreaterEqual(sum(1 for asset in by_room["dining"]
                                        if asset == "builtin:dining_chair"), 2)

    def test_fixture_usage_space_is_kept_clear(self):
        """Nothing blocks a fixture's door swing / pulled-out work space."""
        allowed_in_zone = {"study_desk": ("study_chair",), "dining_chair": ("dining_chair",)}

        def item_of(entry):
            return layout_gen.Item(name=entry["name"], asset=entry["asset"], room=entry["room"],
                                   pos=list(entry["pos"]), rot_z_deg=entry["rot_z_deg"], z=None,
                                   mount="floor", size=None, dims=tuple(entry["size_m"]),
                                   placement=str(entry.get("placement") or ""))

        for shape in ("compact_2x2", "central_living", "corridor_double"):
            scene = layout_gen.generate(1, None, ASSETS_INDEX, verbose=False, shape=shape)
            entries = scene.report["furniture_slots"]
            items = [item_of(entry) for entry in entries]
            for item, entry in zip(items, entries):
                usage = entry.get("usage_m") or 0.0
                if usage <= 0.0:
                    continue
                allow = allowed_in_zone.get(entry["slot"], ())
                zone = layout_gen.usage_rect(item, usage, behind=bool(entry.get("usage_behind")))
                for other, other_entry in zip(items, entries):
                    if other is item or other_entry["slot"] in allow:
                        continue
                    if not layout_gen._blocks_walking(other):
                        continue
                    if checks_mod.rect_distance(other.aabb, item.aabb) < layout_gen.USAGE_NEIGHBOUR_M:
                        continue
                    with self.subTest(shape=shape, fixture=item.name, blocker=other.name):
                        self.assertGreaterEqual(checks_mod.rect_distance(other.aabb, zone),
                                                layout_gen.USAGE_GAP_M)

    def test_fixture_kinds_are_known_to_the_builder(self):
        """run_build dispatches ``builtin:<kind>`` through the same registry."""
        from blender_agent import fixtures
        for asset in fixtures.BUILTIN_DIMS_M:
            self.assertTrue(fixtures.is_fixture(asset.split(":", 1)[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
