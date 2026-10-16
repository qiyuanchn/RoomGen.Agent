"""CLI-level integration: the main validator must measure door **connectors**.

The regression pinned here: ``tools/validate_blender_agent_p0.py``'s ``main()``
used to skip ``door_openings_geometry`` whenever ``spec.openings`` was empty,
so a schema_version 2 spec whose doors all live in ``connectors`` reported
``pass: true, skipped: "spec has no openings"`` while the shell really had door
holes to measure (``outputs/two_floor_demo``: six connections, skipped check).
The underlying :func:`check_door_openings` already walked
``spec.door_connections()`` -- only the entry gate looked at ``openings`` alone.

These tests drive the real CLI entry (:func:`validator.main`: argv, the report
file it writes, the exit code it returns) against synthetic but fully
consistent scene packages -- a shell PLY with real door holes, a manifest, and
depth maps cast through the very mesh the shell holds -- so the skip branch
cannot come back unnoticed:

* a connectors-only spec (no ``openings`` key at all) measures every door, and
  the whole run passes;
* sealing one door in the shell fails ``door_openings_geometry`` and the whole
  run, while every other check (depth, stairs, storey sandwich) still passes;
* the legacy ``openings`` spelling keeps working unchanged, and an equivalent
  connector declaration measures the same geometry to the same numbers;
* a spec with no door connection at all still skips, with an explicit reason.

The runs are rendered at :data:`PANO_SIZE` instead of 2048x1024: the validator's
canonical size is the module constants it reads at call time, and these tests
patch them for the duration of a run (the batch tooling already grades
sub-canonical renders, ``tests/eval_validation_report.py``).  A 2048x1024
cast costs minutes of CPU per camera on a shared, quota-limited host and adds
nothing to the door-connector contract under test.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh
from PIL import Image, ImageDraw

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
for _path in (REPO_ROOT, os.path.join(REPO_ROOT, "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import validate_blender_agent_p0 as validator  # noqa: E402
from blender_agent import export_scene  # noqa: E402
from blender_agent import spec as spec_mod  # noqa: E402

#: ERP size the CLI runs are rendered and verified at (width, height)
PANO_SIZE = (64, 32)

WALL_HALF = 0.06
DOOR_WIDTH = 0.9
DOOR_HEIGHT = 2.05
ROOM_POLYGONS = {
    "a": [[-4.0, -3.0], [0.0, -3.0], [0.0, 3.0], [-4.0, 3.0]],
    "b": [[0.0, -3.0], [4.0, -3.0], [4.0, 3.0], [0.0, 3.0]],
}
STAIR_HOLE = (-1.5, -2.05, -0.5, 2.05)  # x0, y0, x1, y1 of the stairwell

#: the whole building as connectors only -- no legacy ``openings`` key anywhere
TWO_FLOOR_SPEC = {
    "schema_version": 2,
    "unit": "meter",
    "wall_height": 2.7,
    "wall_thickness": 0.12,
    "floors": [{"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15},
               {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15}],
    "rooms": [
        {"id": "a", "type": "room", "floor_id": "f0", "polygon": ROOM_POLYGONS["a"]},
        {"id": "b", "type": "room", "floor_id": "f0", "polygon": ROOM_POLYGONS["b"]},
        {"id": "c", "type": "room", "floor_id": "f1", "polygon": ROOM_POLYGONS["a"]},
        {"id": "d", "type": "room", "floor_id": "f1", "polygon": ROOM_POLYGONS["b"]},
    ],
    "connectors": [
        {"kind": "door", "from_room": "a", "to_room": "b", "from_floor": "f0", "to_floor": "f0",
         "center": [0.0, 0.0], "width": DOOR_WIDTH, "height": DOOR_HEIGHT},
        {"kind": "door", "from_room": "c", "to_room": "d", "from_floor": "f1", "to_floor": "f1",
         "center": [0.0, 0.0], "width": DOOR_WIDTH, "height": DOOR_HEIGHT},
        {"kind": "stair", "from_room": "a", "to_room": "c", "from_floor": "f0", "to_floor": "f1",
         "width": 0.85, "run": 4.0, "rise": 2.85, "steps": 16, "center": [-1.0, 0.0],
         "direction_deg": 90},
    ],
    "slab_openings": [{"outline": [[STAIR_HOLE[0], STAIR_HOLE[1]], [STAIR_HOLE[2], STAIR_HOLE[1]],
                                   [STAIR_HOLE[2], STAIR_HOLE[3]], [STAIR_HOLE[0], STAIR_HOLE[3]]],
                       "floors": ["f1"]}],
    "cameras": [{"name": "0000", "room": "a", "pos": [-2.0, 2.0, 1.5]},
                {"name": "0100", "room": "c", "pos": [-2.0, 2.5, 4.35]}],
}

#: the same two rooms and the same door as a legacy single-storey spec
LEGACY_SPEC = {
    "unit": "meter",
    "wall_height": 2.7,
    "wall_thickness": 0.12,
    "rooms": [{"id": "a", "type": "room", "polygon": ROOM_POLYGONS["a"]},
              {"id": "b", "type": "room", "polygon": ROOM_POLYGONS["b"]}],
    "openings": [{"kind": "door", "between": ["a", "b"], "center": [0.0, 0.0],
                  "width": DOOR_WIDTH, "height": DOOR_HEIGHT}],
    "cameras": [{"name": "0000", "room": "a", "pos": [-2.0, 2.0, 1.5]}],
}

#: the legacy spec with the door connectors-only and a one-storey ``floors`` list
ONE_FLOOR_CONNECTORS_SPEC = {
    "schema_version": 2,
    "unit": "meter",
    "wall_height": 2.7,
    "wall_thickness": 0.12,
    "floors": [{"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15}],
    "rooms": [{"id": "a", "type": "room", "floor_id": "f0", "polygon": ROOM_POLYGONS["a"]},
              {"id": "b", "type": "room", "floor_id": "f0", "polygon": ROOM_POLYGONS["b"]}],
    "connectors": [{"kind": "door", "from_room": "a", "to_room": "b", "from_floor": "f0",
                    "to_floor": "f0", "center": [0.0, 0.0], "width": DOOR_WIDTH,
                    "height": DOOR_HEIGHT}],
    "cameras": [{"name": "0000", "room": "a", "pos": [-2.0, 2.0, 1.5]}],
}


#: the legacy flat with the door 5 cm from the wall end (centre y = 2.5, the
#: span is y in [-3, 3]) -- the same "too close to the corner" shape the
#: shipped two-storey fixture has
CORNER_SPEC = {
    "unit": "meter",
    "wall_height": 2.7,
    "wall_thickness": 0.12,
    "rooms": [{"id": "a", "type": "room", "polygon": ROOM_POLYGONS["a"]},
              {"id": "b", "type": "room", "polygon": ROOM_POLYGONS["b"]}],
    "openings": [{"kind": "door", "between": ["a", "b"], "center": [0.0, 2.5],
                  "width": DOOR_WIDTH, "height": DOOR_HEIGHT}],
    "cameras": [{"name": "0000", "room": "a", "pos": [-2.0, 0.0, 1.5]}],
}


def _box(x0, y0, x1, y1, z0, z1) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0)).apply_translation(
        ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0))


def _shared_wall(z0, z1, door_center=0.0, sealed=False) -> list:
    """The x = 0 wall with a 0.9 x 2.05 m door at ``door_center`` (or sealed shut)."""
    if sealed:
        return [_box(-WALL_HALF, -3.0, WALL_HALF, 3.0, z0, z1)]
    low, high = door_center - DOOR_WIDTH / 2.0, door_center + DOOR_WIDTH / 2.0
    return [
        _box(-WALL_HALF, -3.0, WALL_HALF, low, z0, z1),
        _box(-WALL_HALF, high, WALL_HALF, 3.0, z0, z1),
        _box(-WALL_HALF, low, WALL_HALF, high, z0 + DOOR_HEIGHT, z1),
    ]


def _storey_shell(z0, z1, door_center=0.0, sealed=False) -> list:
    """One storey: the shared wall plus the four enclosing walls."""
    return _shared_wall(z0, z1, door_center=door_center, sealed=sealed) + [
        _box(-4.0, -3.06, 4.0, -2.94, z0, z1),
        _box(-4.0, 2.94, 4.0, 3.06, z0, z1),
        _box(-4.06, -3.06, -3.94, 3.06, z0, z1),
        _box(3.94, -3.06, 4.06, 3.06, z0, z1),
    ]


def _upper_slab() -> list:
    """The floor f1 slab at 2.70..2.85 with the stairwell cut out."""
    x0, y0, x1, y1 = STAIR_HOLE
    return [
        _box(-4.0, -3.0, x0, 3.0, 2.7, 2.85),
        _box(x1, -3.0, 4.0, 3.0, 2.7, 2.85),
        _box(x0, -3.0, x1, y0, 2.7, 2.85),
        _box(x0, y1, x1, 3.0, 2.7, 2.85),
    ]


def _dome() -> trimesh.Trimesh:
    """The far enclosure, as in a real build (the caster handles it analytically)."""
    return trimesh.creation.icosphere(subdivisions=3, radius=30.0).apply_translation(
        (0.0, 0.0, 2.7))


def _two_floor_mesh(sealed_lower_door: bool = False) -> trimesh.Trimesh:
    spec = spec_mod.parse_spec(json.loads(json.dumps(TWO_FLOOR_SPEC)))
    boxes = _storey_shell(0.0, 2.7, sealed=sealed_lower_door)
    boxes += _storey_shell(2.85, 5.55)
    boxes.append(_box(-4.0, -3.0, 4.0, 3.0, -0.15, 0.0))  # ground slab
    boxes += _upper_slab()
    for step in spec.stair_plans()[0].boxes:
        xs = [point[0] for point in step["polygon"]]
        ys = [point[1] for point in step["polygon"]]
        boxes.append(_box(min(xs), min(ys), max(xs), max(ys), step["z0"], step["z1"]))
    boxes.append(_dome())
    return trimesh.util.concatenate(boxes)


def _legacy_mesh(sealed: bool = False, door_center: float = 0.0) -> trimesh.Trimesh:
    boxes = _storey_shell(0.0, 2.7, door_center=door_center, sealed=sealed)
    boxes.append(_box(-4.0, -3.0, 4.0, 3.0, -0.15, 0.0))
    boxes.append(_dome())
    return trimesh.util.concatenate(boxes)


def _write_topdown(scene_dir: str) -> None:
    image = Image.new("L", (200, 150), 0)
    draw = ImageDraw.Draw(image)
    for room in (ROOM_POLYGONS["a"], ROOM_POLYGONS["b"]):
        draw.polygon([(10 + (x + 4.0) * 22.0, 10 + (y + 3.0) * 21.0) for x, y in room],
                     fill=120, outline=255)
    image.save(os.path.join(scene_dir, "topdown.png"))


def _write_package(root: str, spec_data: dict, mesh: trimesh.Trimesh, size=PANO_SIZE) -> tuple:
    """A self-consistent build+scene package on disk: shell, manifest, viewpoints.

    The depth maps are cast with the validator's own :func:`erp_rays` through the
    very mesh the shell PLY holds, so ``depth_matches_geometry`` sees an exact
    render and the run's verdict turns on the check under test instead of on a
    synthetic depth image.
    """
    width, height = size
    build_dir = os.path.join(root, "build")
    scene_dir = os.path.join(root, "scene")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "viewpoints"), exist_ok=True)
    mesh.export(os.path.join(build_dir, "shell.ply"))
    with open(os.path.join(build_dir, "build_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"render": {"width": width, "height": height},
                   "visible_geometry": {"ply": "shell.ply"},
                   "shell": {"ply": "shell.ply"},
                   "depth_scale": spec_mod.DEPTH_SCALE}, handle)
    spec_path = os.path.join(root, "spec.json")
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(spec_data, handle)

    spec = spec_mod.load_spec(spec_path)
    caster = validator.MeshRayCaster(mesh)
    for camera in spec.cameras:
        view_dir = os.path.join(scene_dir, "viewpoints", camera.name)
        os.makedirs(view_dir, exist_ok=True)
        np.savetxt(os.path.join(view_dir, "extrinsics.txt"),
                   np.array(spec_mod.extrinsics_matrix(camera.pos, 0.0), dtype=np.float64),
                   fmt="%.10f")
        origins, directions = validator.erp_rays(
            np.array(spec_mod.extrinsics_matrix(camera.pos, 0.0), dtype=np.float64), width, height)
        depth = caster.cast(origins, directions).reshape(height, width)
        depth[~np.isfinite(depth)] = 0.0
        Image.fromarray(export_scene.encode_depth(depth, spec_mod.DEPTH_SCALE)).save(
            os.path.join(view_dir, "place_depth.png"))
        with open(os.path.join(view_dir, "place_depth_scale.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"{spec_mod.DEPTH_SCALE:g}\n")
        Image.new("RGB", (width, height), (24, 24, 24)).save(
            os.path.join(view_dir, "panoImage_2048.png"))
    with open(os.path.join(scene_dir, "map.json"), "w", encoding="utf-8") as handle:
        json.dump(spec.camera_adjacency(), handle)
    _write_topdown(scene_dir)
    return spec_path, build_dir, scene_dir


def _run_cli(root: str, spec_path: str, build_dir: str, scene_dir: str) -> tuple:
    """The real CLI: argv in, exit code and report file out."""
    report_path = os.path.join(root, "validation_report.json")
    exit_code = validator.main(["--build", build_dir, "--scene", scene_dir, "--spec", spec_path,
                                "--report", report_path, "--sample-rays", "256"])
    with open(report_path, "r", encoding="utf-8") as handle:
        return exit_code, json.load(handle)


def _check(report: dict, name: str) -> dict:
    for item in report["checks"]:
        if item["name"] == name:
            return item
    raise AssertionError(f"the report has no '{name}' check: "
                         f"{[item['name'] for item in report['checks']]}")


def _failed(report: dict) -> list:
    return [item["name"] for item in report["checks"] if not item["pass"]]


def _measurement(door: dict) -> dict:
    """The resolution-independent numbers of one door probe report."""
    return {
        "pass": door["pass"],
        "width": door["measured_width_m"],
        "height": door["measured_height_m"],
        "sides": {room: [side["open_fraction"], side["wall_solid_fraction"],
                         side["open_width_m"], side["open_height_m"]]
                  for room, side in door.get("sides", {}).items()},
    }


class ConnectorDoorGateTest(unittest.TestCase):
    """The CLI gate on doors declared in ``connectors`` (and in ``openings``)."""

    def setUp(self):
        #: the validator's canonical size is what these runs are rendered at;
        #: see the module docstring for why 2048x1024 is not used here
        self._canonical = (spec_mod.PANO_WIDTH, spec_mod.PANO_HEIGHT)
        spec_mod.PANO_WIDTH, spec_mod.PANO_HEIGHT = PANO_SIZE

    def tearDown(self):
        spec_mod.PANO_WIDTH, spec_mod.PANO_HEIGHT = self._canonical

    def test_connectors_only_spec_measures_every_door(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec_path, build_dir, scene_dir = _write_package(
                tmp, TWO_FLOOR_SPEC, _two_floor_mesh())
            spec = spec_mod.load_spec(spec_path)
            self.assertEqual(spec.openings, [])  # nothing to fall back on
            self.assertEqual(len(spec.door_connections()), 2)
            exit_code, report = _run_cli(tmp, spec_path, build_dir, scene_dir)
            failed = _failed(report)

        self.assertEqual(exit_code, 0, f"the full validation failed: {failed}")
        self.assertTrue(report["pass"], failed)
        entry = _check(report, "door_openings_geometry")
        self.assertNotIn("skipped", entry, "a connectors-only spec must not skip the door check")
        self.assertEqual(entry["openings"], len(spec.door_connections()))
        self.assertTrue(entry["pass"], entry["doors"])
        self.assertEqual([door["between"] for door in entry["doors"]], [["a", "b"], ["c", "d"]])
        self.assertEqual([door["floor"] for door in entry["doors"]], ["f0", "f1"])
        for door in entry["doors"]:
            self.assertTrue(door["pass"], door["failures"])
            self.assertAlmostEqual(door["measured_width_m"], DOOR_WIDTH, delta=0.06)
            self.assertAlmostEqual(door["measured_height_m"], DOOR_HEIGHT, delta=0.08)

    def test_sealing_one_door_fails_the_whole_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec_path, build_dir, scene_dir = _write_package(
                tmp, TWO_FLOOR_SPEC, _two_floor_mesh(sealed_lower_door=True))
            exit_code, report = _run_cli(tmp, spec_path, build_dir, scene_dir)
            failed = _failed(report)
            entry = _check(report, "door_openings_geometry")

        self.assertNotEqual(exit_code, 0, "a sealed door must fail the run")
        self.assertFalse(report["pass"])
        self.assertEqual(failed, ["door_openings_geometry"], report["checks"])
        self.assertFalse(entry["pass"])
        sealed = [door for door in entry["doors"] if door["between"] == ["a", "b"]][0]
        self.assertFalse(sealed["pass"])
        self.assertTrue(any("not open" in failure or "measured opening width" in failure
                            for failure in sealed["failures"]), sealed["failures"])
        # only the door contract broke: the rest of the shell is untouched
        self.assertTrue(_check(report, "depth_matches_geometry")["pass"])
        self.assertTrue(_check(report, "stairs_built")["pass"])
        self.assertTrue(_check(report, "floor_stacks")["pass"])

    def test_legacy_openings_are_measured_and_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec_path, build_dir, scene_dir = _write_package(tmp, LEGACY_SPEC, _legacy_mesh())
            spec = spec_mod.load_spec(spec_path)
            self.assertEqual(spec.openings and len(spec.openings), 1)
            exit_code, report = _run_cli(tmp, spec_path, build_dir, scene_dir)
            failed = _failed(report)

        self.assertEqual(exit_code, 0, f"the legacy openings run failed: {failed}")
        self.assertTrue(report["pass"], failed)
        entry = _check(report, "door_openings_geometry")
        self.assertNotIn("skipped", entry)
        self.assertEqual(entry["openings"], 1)
        self.assertTrue(entry["pass"], entry["doors"])
        self.assertAlmostEqual(entry["doors"][0]["measured_width_m"], DOOR_WIDTH, delta=0.06)

    def test_equivalent_connectors_measure_the_same_as_openings(self):
        """The merged list: the same door in either spelling, the same numbers."""
        mesh = _legacy_mesh()
        caster = validator.MeshRayCaster(mesh)
        legacy = validator.check_door_openings(spec_mod.parse_spec(LEGACY_SPEC), caster)
        connectors = validator.check_door_openings(
            spec_mod.parse_spec(ONE_FLOOR_CONNECTORS_SPEC), caster)
        self.assertTrue(legacy["pass"], legacy["doors"])
        self.assertTrue(connectors["pass"], connectors["doors"])
        self.assertEqual(legacy["openings"], connectors["openings"])
        self.assertEqual(_measurement(legacy["doors"][0]), _measurement(connectors["doors"][0]))

    def test_no_connections_still_skips_with_a_reason(self):
        spec_data = dict(LEGACY_SPEC)
        spec_data["openings"] = []
        with tempfile.TemporaryDirectory() as tmp:
            spec_path, build_dir, scene_dir = _write_package(tmp, spec_data, _legacy_mesh())
            exit_code, report = _run_cli(tmp, spec_path, build_dir, scene_dir)
            failed = _failed(report)
            entry = _check(report, "door_openings_geometry")

        self.assertEqual(exit_code, 0, f"a doorless spec must not fail: {failed}")
        self.assertTrue(report["pass"], failed)
        self.assertTrue(entry["pass"])
        self.assertIn("skipped", entry)
        self.assertIn("no door connections", entry["skipped"])


class DoorNearWallEndTest(unittest.TestCase):
    """A door close to a wall end is probed where its wall span really is.

    ``tests/fixtures/two_floor_demo.json`` has one: centre 0.5 m from the wall
    corner, so the jamb is 5 cm from the span end.  The nominal "wall beside the
    opening" probe stands 0.15 m past the jamb -- past the end of the shared
    span, i.e. outside the wall -- and read the empty space there as "the wall
    is missing next to the door".  With ``door_openings_geometry`` now really
    gating connectors, that would fail the shipped fixture and any hand-written
    scene with a corner door, so the probe is pulled back inside the span (and
    the width scan with it) instead of sampling past the wall.
    """

    def test_the_wall_beside_a_corner_door_is_probed_inside_the_span(self):
        spec = spec_mod.parse_spec(CORNER_SPEC)
        report = validator.check_door_openings(
            spec, validator.MeshRayCaster(_legacy_mesh(door_center=2.5)))
        self.assertTrue(report["pass"], report["doors"][0].get("failures"))
        door = report["doors"][0]
        self.assertAlmostEqual(door["measured_width_m"], DOOR_WIDTH, delta=0.06)
        self.assertAlmostEqual(door["measured_height_m"], DOOR_HEIGHT, delta=0.08)
        for room_id, side in door["sides"].items():
            self.assertEqual(side["wall_beside_probes"], 2, room_id)
            self.assertEqual(side["wall_solid_fraction"], 1.0, room_id)

    def test_a_missing_wall_sliver_beside_a_corner_door_still_fails(self):
        """The clamp keeps the gate: no wall at all beside the door fails it."""
        spec = spec_mod.parse_spec(CORNER_SPEC)
        # the wall spans the door and one jamb only: the sliver past y = 2.95
        # is missing, exactly where the clamped probe stands
        boxes = [
            _box(-WALL_HALF, -3.0, WALL_HALF, 2.05, 0.0, 2.7),
            _box(-WALL_HALF, 2.05, WALL_HALF, 2.95, 2.05, 2.7),
            _box(-4.0, -3.0, 4.0, 3.0, -0.15, 0.0),
            _dome(),
        ]
        report = validator.check_door_openings(
            spec, validator.MeshRayCaster(trimesh.util.concatenate(boxes)))
        self.assertFalse(report["pass"])
        failures = " ".join(report["doors"][0]["failures"])
        self.assertIn("wall is missing next to the door", failures)


if __name__ == "__main__":
    unittest.main(verbosity=2)
