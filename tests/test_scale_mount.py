"""Scale and mounting-height consistency between the checker and the builder.

Run::

    python tests/test_scale_mount.py          # or: python -m unittest tests.test_scale_mount

Both facts below were the third review's counterexamples and are frozen here:

* **scale**: ``checks.item_dims`` multiplies a ``builtin:`` item's measured
  footprint by ``FurnitureSpec.scale`` while the builder ignored it, so two
  wardrobes authored at ``scale = 0.25`` were checked as 0.30 m wide and built
  1.20 m wide -- overlapping by about 0.70 m in a scene the checker passed.
  The generator now scales the whole part around its own origin
  (``fixtures.world_boxes``) and the manifest carries the scaled numbers, so
  the checked box *is* the built one.  The Blender-backed test measures the real
  objects and compares them with ``checks.item_boxes``.
* **mounting height**: on a non-zero storey elevation a ``mount: "ceiling"``
  item was offset by the floor height twice (storey 2.85 m, room top 5.55 m, a
  0.40 m pendant landed at 8.00..8.40 m instead of 5.15..5.55 m).  Both sides
  now resolve the world z with ``checks.item_world_z``, and the builder hangs
  ceiling-mounted ``builtin:`` parts there too instead of leaving them on the
  floor.

The Blender half is skipped when no Blender binary is installed (same rule as
``test_settle.py``); everything else runs on the host python.

The asset-orientation half freezes the fourth review's counterexample: the
Poly Haven registry measures ``modern_coffee_table_01`` as 1.20 x 0.60 m but
the glTF imports as 0.60 x 1.20 m (X/Y transposed), so the checker and the
builder disagreed about the footprint at every yaw.  ``normalise_asset`` now
aligns a transposed footprint to ``api_dimensions_mm`` with one quarter turn,
and its ``yaw_deg`` front correction actually moves the geometry (glTF imports
are in quaternion mode; the old code wrote the ignored Euler channels).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import assets as assets_mod  # noqa: E402
from blender_agent import checks as checks_mod  # noqa: E402
from blender_agent import fixtures as fixtures_mod  # noqa: E402
from blender_agent import spec as spec_mod  # noqa: E402

ELEVATION_M = 2.85          # the second storey of the review's repro
ROOM_HEIGHT_M = 2.7
ROOM_TOP_M = ELEVATION_M + ROOM_HEIGHT_M             # 5.55
PENDANT_HEIGHT_M = 0.40                             # the review's pendant
PENDANT_INDEX = {"pendant_040": {"dims_m": (0.30, 0.30, PENDANT_HEIGHT_M),
                                 "categories": ["lamp"], "attributes": {}, "ref": None,
                                 "source": "test"}}

#: the review's repro: two wardrobes at ``scale = 0.25``, half a metre apart
WARDROBE_SCALE = 0.25
WARDROBE_SPOTS = ([2.0, 0.4], [2.5, 0.4])


def blender_binary():
    """The Blender used by the runner scripts, or ``None`` when not installed."""
    candidate = os.environ.get("PANOWORLD_BLENDER") or os.path.join(
        REPO_ROOT, "third_party", "blender-4.5.14-linux-x64", "blender")
    return candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else None


def spec_dict(elevation: float = 0.0, furniture=None, height: float = ROOM_HEIGHT_M,
              polygon=None) -> dict:
    """One 10x10 m room on one storey -- the review's repro stage."""
    polygon = polygon or [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    return {
        "schema_version": 2,
        "wall_height": height,
        "wall_thickness": 0.12,
        "floors": [{"id": "f1", "elevation": elevation, "height": height,
                    "slab_thickness": 0.15}],
        "rooms": [{"id": "room", "type": "living_room", "floor_id": "f1", "polygon": polygon}],
        "cameras": [{"name": "0000", "room": "room", "pos": [8.5, 8.5, elevation + 1.5]}],
        "furniture": list(furniture or []),
    }


def parse(spec_dict_value: dict):
    return spec_mod.parse_spec(spec_dict_value)


def union_aabb(boxes) -> tuple:
    """World ``(x0, y0, x1, y1)``, ``(z0, z1)`` of a ``fixtures.world_boxes`` list."""
    xs, ys, z0s, z1s = [], [], [], []
    for box in boxes:
        cx, cy = box["center"]
        dx, dy = box["direction"]
        width, depth, _height = box["size"]
        nx, ny = -dy, dx
        for along in (-1.0, 1.0):
            for across in (-1.0, 1.0):
                xs.append(cx + dx * (width / 2.0) * along + nx * (depth / 2.0) * across)
                ys.append(cy + dy * (width / 2.0) * along + ny * (depth / 2.0) * across)
        z0s.append(box["z0"])
        z1s.append(box["z1"])
    return (min(xs), min(ys), max(xs), max(ys)), (min(z0s), max(z1s))


class BuiltinScaleGeometryTest(unittest.TestCase):
    """Every fixture part scales around its own origin, checker included."""

    def test_world_boxes_match_the_checked_footprint(self):
        for part in sorted(fixtures_mod.FIXTURE_PARTS):
            for scale in (0.25, 0.5, 1.0, 1.75):
                for rot in (0.0, 90.0, 180.0, 270.0):
                    with self.subTest(part=part, scale=scale, rot=rot):
                        boxes = fixtures_mod.world_boxes(part, (3.0, 1.5), rot, base_z=1.1,
                                                         scale=scale)
                        self.assertTrue(boxes)
                        aabb, z = union_aabb(boxes)
                        dims = checks_mod.item_dims(f"builtin:{part}", {}, None, scale)
                        reference = checks_mod.rotated_aabb((3.0, 1.5), dims, rot)
                        for measured, expected in zip(aabb, reference):
                            self.assertAlmostEqual(measured, expected, places=9)
                        self.assertAlmostEqual(z[0], 1.1, places=9)
                        self.assertAlmostEqual(z[1], 1.1 + dims[2], places=9)

    def test_dims_and_usage_scale_with_the_part(self):
        for part in sorted(fixtures_mod.FIXTURE_PARTS):
            base = fixtures_mod.FIXTURE_PARTS[part]
            self.assertEqual(fixtures_mod.dims(part), base["dims"])
            self.assertEqual(fixtures_mod.usage_m(part), base["usage_m"])
            for scale in (0.25, 1.0, 2.0):
                self.assertEqual(fixtures_mod.dims(part, scale),
                                 tuple(v * scale for v in base["dims"]))
                self.assertAlmostEqual(fixtures_mod.usage_m(part, scale),
                                       base["usage_m"] * scale, places=12)

    def test_rug_and_lamp_footprints_scale_too(self):
        rug = checks_mod.item_dims("builtin:rug", {}, [2.0, 1.5], 0.5)
        self.assertEqual(rug, (1.0, 0.75, checks_mod.RUG_THICKNESS_M))
        lamp = checks_mod.BUILTIN_DIMS_M["builtin:table_lamp"]
        self.assertEqual(checks_mod.item_dims("builtin:table_lamp", {}, None, 0.5),
                         tuple(v * 0.5 for v in lamp))


class WardrobeScaleCounterexampleTest(unittest.TestCase):
    """The review's problem A: checked size 0.30 m, built size 1.20 m."""

    def counterexample_spec(self, scale: float):
        items = [{"asset": "builtin:wardrobe", "room": "room", "pos": list(spot),
                  "rot_z_deg": 0, "scale": scale, "name": name}
                 for name, spot in zip(("w_a", "w_b"), WARDROBE_SPOTS)]
        return parse(spec_dict(0.0, items))

    def test_checked_sizes_are_the_built_sizes(self):
        spec = self.counterexample_spec(WARDROBE_SCALE)
        boxes, unknown = checks_mod.item_boxes(spec, {})
        self.assertEqual(unknown, [])
        self.assertEqual([box.name for box in boxes], ["w_a", "w_b"])
        for box, spot in zip(boxes, WARDROBE_SPOTS):
            generated = fixtures_mod.world_boxes("wardrobe", spot, 0.0,
                                                 scale=WARDROBE_SCALE)
            aabb, z = union_aabb(generated)
            self.assertAlmostEqual(aabb[0], box.aabb[0], places=9)
            self.assertAlmostEqual(aabb[2], box.aabb[2], places=9)
            self.assertAlmostEqual(z[0], box.z[0], places=9)
            self.assertAlmostEqual(z[1], box.z[1], places=9)
        # the review's table: 0.30 m wide, 0.525 m tall, X 1.85..2.15 / 2.35..2.65
        self.assertAlmostEqual(boxes[0].dims[0], 0.30, places=6)
        self.assertAlmostEqual(boxes[0].dims[2], 0.525, places=6)
        self.assertAlmostEqual(boxes[0].aabb[0], 1.85, places=6)
        self.assertAlmostEqual(boxes[0].aabb[2], 2.15, places=6)
        self.assertAlmostEqual(boxes[1].aabb[0], 2.35, places=6)
        self.assertAlmostEqual(boxes[1].aabb[2], 2.65, places=6)
        # ... and the built geometry leaves the 0.20 m gap between them
        self.assertLess(boxes[0].aabb[2], boxes[1].aabb[0])
        report = checks_mod.check_spec(spec, {})
        self.assertTrue(report["pass"], report["violations"])

    def test_the_same_layout_at_scale_one_is_still_caught(self):
        """The check is not weakened: at 1:1 the two wardrobes really do overlap."""
        spec = self.counterexample_spec(1.0)
        report = checks_mod.check_spec(spec, {})
        self.assertFalse(report["pass"])
        self.assertTrue(any("overlaps" in violation for violation in report["violations"]),
                        report["violations"])
        boxes, _unknown = checks_mod.item_boxes(spec, {})
        self.assertGreater(boxes[0].aabb[2] - boxes[1].aabb[0], 0.6)  # the 0.70 m of the review
        generated_a, _z = union_aabb(fixtures_mod.world_boxes("wardrobe", WARDROBE_SPOTS[0], 0.0))
        generated_b, _z = union_aabb(fixtures_mod.world_boxes("wardrobe", WARDROBE_SPOTS[1], 0.0))
        self.assertAlmostEqual(generated_a[2], boxes[0].aabb[2], places=9)
        self.assertAlmostEqual(generated_b[0], boxes[1].aabb[0], places=9)


class CeilingMountHeightTest(unittest.TestCase):
    """The review's problem B: the storey elevation was added twice."""

    def pendant_spec(self, elevation: float):
        item = {"asset": "ph:pendant_040", "room": "room", "pos": [5.0, 5.0],
                "rot_z_deg": 0, "mount": "ceiling", "name": "pendant"}
        return parse(spec_dict(elevation, [item]))

    def test_pendant_hangs_from_its_own_room_ceiling(self):
        spec = self.pendant_spec(ELEVATION_M)
        boxes, _unknown = checks_mod.item_boxes(spec, PENDANT_INDEX)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].z[1], ROOM_TOP_M)
        self.assertAlmostEqual(boxes[0].z[0], ROOM_TOP_M - PENDANT_HEIGHT_M, places=9)
        # the pre-fix numbers (elevation added a second time) must never come back
        self.assertLess(boxes[0].z[0], ROOM_TOP_M - PENDANT_HEIGHT_M + 1e-6)
        self.assertEqual(boxes[0].placement, checks_mod.CEILING)

    def test_ground_floor_ceiling_item_is_unchanged(self):
        spec = self.pendant_spec(0.0)
        boxes, _unknown = checks_mod.item_boxes(spec, PENDANT_INDEX)
        self.assertAlmostEqual(boxes[0].z[0], ROOM_HEIGHT_M - PENDANT_HEIGHT_M, places=9)
        self.assertAlmostEqual(boxes[0].z[1], ROOM_HEIGHT_M, places=9)

    def test_item_world_z_is_the_single_rule(self):
        spec = self.pendant_spec(ELEVATION_M)
        item = spec.furniture[0]
        floor_z, bottom, top = checks_mod.item_world_z(spec, item, (0.30, 0.30, PENDANT_HEIGHT_M))
        self.assertEqual(floor_z, ELEVATION_M)
        self.assertAlmostEqual(bottom, 5.15, places=9)
        self.assertAlmostEqual(top, 5.55, places=9)
        # a ceiling mount wins over an authored world z, a floor item uses it
        item.z = 4.0
        self.assertAlmostEqual(checks_mod.item_world_z(spec, item, (0.3, 0.3, 0.4))[1], 5.15, places=9)
        item.mount = "floor"
        self.assertAlmostEqual(checks_mod.item_world_z(spec, item, (0.3, 0.3, 0.4))[1], 4.0, places=9)
        item.z = None
        self.assertAlmostEqual(checks_mod.item_world_z(spec, item, (0.3, 0.3, 0.4))[1],
                               ELEVATION_M, places=9)

    def test_builtin_part_scales_and_hangs(self):
        """A 0.70 m upper cabinet at 4/7 scale is the review's 0.40 m pendant."""
        item = {"asset": "builtin:kitchen_upper", "room": "room", "pos": [5.0, 5.0],
                "rot_z_deg": 0, "mount": "ceiling", "scale": 4.0 / 7.0, "name": "upper"}
        spec = parse(spec_dict(ELEVATION_M, [item]))
        boxes, unknown = checks_mod.item_boxes(spec, {})
        self.assertEqual(unknown, [])
        self.assertAlmostEqual(boxes[0].dims[2], PENDANT_HEIGHT_M, places=9)
        self.assertAlmostEqual(boxes[0].z[0], ROOM_TOP_M - PENDANT_HEIGHT_M, places=9)
        self.assertAlmostEqual(boxes[0].z[1], ROOM_TOP_M, places=9)
        generated = fixtures_mod.world_boxes("kitchen_upper", (5.0, 5.0), 0.0,
                                             base_z=boxes[0].z[0], scale=4.0 / 7.0)
        _aabb, z = union_aabb(generated)
        self.assertAlmostEqual(z[0], boxes[0].z[0], places=9)
        self.assertAlmostEqual(z[1], boxes[0].z[1], places=9)

    def test_wall_mounted_item_keeps_its_authored_height(self):
        item = {"asset": "builtin:kitchen_upper", "room": "room", "pos": [3.0, 0.3],
                "rot_z_deg": 0, "z": ELEVATION_M + 1.45, "mount": "wall", "name": "upper"}
        spec = parse(spec_dict(ELEVATION_M, [item]))
        boxes, _unknown = checks_mod.item_boxes(spec, {})
        self.assertAlmostEqual(boxes[0].z[0], ELEVATION_M + 1.45, places=9)
        self.assertEqual(boxes[0].placement, checks_mod.MOUNTED)


# --------------------------------------------------------------------------- #
# the real builder: measure the geometry Blender emits
# --------------------------------------------------------------------------- #
#: runs ``run_build`` on the spec and writes the *measured* world AABB of every
#: furniture placement (its recorded objects, or the props built by name)
MEASURE_SCRIPT = '''
import json, os, sys

REPO_ROOT = os.environ["ROOMGEN_REPO_ROOT"]
sys.path.insert(0, REPO_ROOT)

import bpy
from blender_agent import run_build


def world_aabb(objects):
    xs, ys, zs = [], [], []
    for obj in objects:
        if obj.type != "MESH":
            continue
        for vertex in obj.data.vertices:
            point = obj.matrix_world @ vertex.co
            xs.append(point.x)
            ys.append(point.y)
            zs.append(point.z)
    if not xs:
        return None
    return [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    spec_path, build_dir = argv[0], argv[1]
    # the measurement targets the authored geometry: no physics stage
    with open(spec_path, "r", encoding="utf-8") as handle:
        block = json.load(handle)
    block.setdefault("settle", {})["enabled"] = False
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(block, handle, indent=2)

    code = run_build.main(["--spec", spec_path, "--out", build_dir, "--skip-render",
                           "--samples", "1", "--width", "64", "--height", "32"])
    if code != 0:
        raise SystemExit(f"run_build exited with {code}")

    with open(os.path.join(build_dir, "build_manifest.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    measured, missing = {}, []
    for placement in manifest["furniture"]:
        names = list(placement.get("objects") or [])
        if not names:
            label = placement["name"]
            if placement["asset"] == "builtin:rug":
                names = [f"rug_{label}"]
            elif placement["asset"] == "builtin:table_lamp":
                names = [obj.name for obj in bpy.data.objects if obj.name.startswith(f"lamp_{label}_")]
            else:
                names = [placement.get("instance") or ""]
        objects = [bpy.data.objects[name] for name in names if name in bpy.data.objects]
        if len(objects) != len(names):
            missing.append({"name": placement["name"], "names": names})
            continue
        aabb = world_aabb(objects)
        if aabb is not None:
            measured[placement["name"]] = aabb
    report = {"measured": measured, "missing": missing,
              "furniture": manifest["furniture"],
              "lighting": manifest["lighting"]}
    with open(os.path.join(build_dir, "measured.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


main()
'''


class BlenderGeneratedGeometryTest(unittest.TestCase):
    """Measure what Blender really built and hold it against the checker."""

    def run_build_and_measure(self, spec_json: dict) -> dict:
        binary = blender_binary()
        if binary is None:
            self.skipTest("no Blender binary (PANOWORLD_BLENDER / third_party/blender-*)")
        with tempfile.TemporaryDirectory() as workdir:
            spec_path = os.path.join(workdir, "spec.json")
            with open(spec_path, "w", encoding="utf-8") as handle:
                json.dump(spec_json, handle, indent=2)
            script_path = os.path.join(workdir, "measure.py")
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(MEASURE_SCRIPT)
            build_dir = os.path.join(workdir, "build")
            environment = dict(os.environ, ROOMGEN_REPO_ROOT=REPO_ROOT)
            result = subprocess.run(
                [binary, "--background", "--factory-startup", "--python", script_path,
                 "--", spec_path, build_dir],
                capture_output=True, text=True, timeout=1800, env=environment,
            )
            measured_path = os.path.join(build_dir, "measured.json")
            self.assertTrue(os.path.isfile(measured_path),
                            f"the build wrote no measurement\n{result.stdout[-3000:]}\n"
                            f"{result.stderr[-3000:]}")
            with open(measured_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertEqual(report["missing"], [])
        return report, spec_mod.parse_spec(spec_json)

    def test_counterexample_scene_measures_as_checked(self):
        furniture = [
            {"asset": "builtin:wardrobe", "room": "room", "pos": [2.0, 0.4], "rot_z_deg": 0,
             "scale": WARDROBE_SCALE, "name": "w_a"},
            {"asset": "builtin:wardrobe", "room": "room", "pos": [2.5, 0.4], "rot_z_deg": 0,
             "scale": WARDROBE_SCALE, "name": "w_b"},
            {"asset": "builtin:kitchen_upper", "room": "room", "pos": [5.0, 5.0], "rot_z_deg": 0,
             "mount": "ceiling", "scale": 4.0 / 7.0, "name": "upper"},
            {"asset": "builtin:table_lamp", "room": "room", "pos": [5.0, 4.0], "rot_z_deg": 0,
             "mount": "ceiling", "scale": 0.5, "name": "pendant_lamp"},
            {"asset": "builtin:rug", "room": "room", "pos": [5.0, 2.0], "rot_z_deg": 0,
             "size": [2.0, 1.5], "scale": 0.5, "name": "rug"},
        ]
        report, spec = self.run_build_and_measure(spec_dict(ELEVATION_M, furniture))
        boxes, unknown = checks_mod.item_boxes(spec, {})
        self.assertEqual(unknown, [])
        self.assertEqual(sorted(report["measured"]), sorted(box.name for box in boxes))
        for box in boxes:
            measured = report["measured"][box.name]
            with self.subTest(item=box.name):
                self.assertAlmostEqual(measured[0], box.aabb[0], places=6)
                self.assertAlmostEqual(measured[1], box.aabb[1], places=6)
                self.assertAlmostEqual(measured[3], box.aabb[2], places=6)
                self.assertAlmostEqual(measured[4], box.aabb[3], places=6)
                self.assertAlmostEqual(measured[2], box.z[0], places=6)
                if box.asset == "builtin:rug":
                    # the built slab is 12 mm; the checked AABB uses the fixed
                    # RUG_THICKNESS_M (0.04 m), a rug never blocks or collides
                    self.assertGreaterEqual(box.z[1] - measured[5], 0.02)
                else:
                    self.assertAlmostEqual(measured[5], box.z[1], places=6)
        # the review's numbers, measured on the real objects
        wardrobe = report["measured"]["w_a"]
        self.assertAlmostEqual(wardrobe[3] - wardrobe[0], 0.30, places=6)
        self.assertAlmostEqual(wardrobe[5] - wardrobe[2], 0.525, places=6)
        self.assertGreater(report["measured"]["w_b"][0] - wardrobe[3], 0.19)
        # the rug's slab carries the scaled size, not the authored [2.0, 1.5]
        rug = report["measured"]["rug"]
        self.assertAlmostEqual(rug[3] - rug[0], 1.0, places=6)
        self.assertAlmostEqual(rug[4] - rug[1], 0.75, places=6)
        # ceiling mounted parts hang from this storey's ceiling, not from 8.0 m
        for name in ("upper", "pendant_lamp"):
            self.assertAlmostEqual(report["measured"][name][5], ROOM_TOP_M, places=6)
        self.assertAlmostEqual(report["measured"]["upper"][2],
                               ROOM_TOP_M - PENDANT_HEIGHT_M, places=6)
        # ... and the manifest carries the same facts
        placements = {entry["name"]: entry for entry in report["furniture"]}
        for name, entry in placements.items():
            measured = report["measured"][name]
            self.assertAlmostEqual(entry["size_m"][0], measured[3] - measured[0], places=3)
            self.assertAlmostEqual(entry["bbox_z"][0], measured[2], places=4)
            self.assertAlmostEqual(entry["size_m"][2], measured[5] - measured[2], places=3)
            self.assertAlmostEqual(entry["bbox_z"][1], measured[5], places=4)
        # a pendant light follows the scaled lamp (0.34 * 0.5 = 0.17 m tall)
        lamp = placements["pendant_lamp"]
        self.assertAlmostEqual(lamp["size_m"][2], 0.17, places=4)
        light = [position for position in report["lighting"]["lamp_positions"]
                 if abs(position[0] - 5.0) < 1e-6 and abs(position[1] - 4.0) < 1e-6]
        self.assertEqual(len(light), 1)
        self.assertAlmostEqual(light[0][2], lamp["bbox_z"][0] + 0.88 * 0.17, places=6)
        self.assertLess(light[0][2], ROOM_TOP_M)


# --------------------------------------------------------------------------- #
# registry frame: the coffee table's glTF is yawed against its own metadata
# --------------------------------------------------------------------------- #
COFFEE_TABLE_GLTF = os.path.join(
    REPO_ROOT, "third_party", "assets", "polyhaven", "models",
    "modern_coffee_table_01", "modern_coffee_table_01_1k.gltf",
)
#: Poly Haven's ``dimensions`` for modern_coffee_table_01, the values the
#: checker reads: width 1.2018 m (X), depth 0.600 m (Y), height 0.390 m (Z)
COFFEE_TABLE_REGISTRY_M = (1.2018, 0.600, 0.390)

ORIENTATION_SCRIPT = '''
import json, os, sys

ROOT = os.environ["ROOMGEN_REPO_ROOT"]
sys.path.insert(0, ROOT)

import bpy
from blender_agent import assets

cache = os.path.join(ROOT, "third_party", "assets", "polyhaven")
entry = assets.load_index(cache)["models"]["modern_coffee_table_01"]

result = {}
bpy.ops.wm.read_factory_settings(use_empty=True)
_collection, meshes, _imported = assets.import_gltf_asset(os.environ["COFFEE_GLTF"], "coffee")
result["coffee"] = assets.normalise_asset(meshes, "modern_coffee_table_01", {}, entry)

# a 2 x 1 x 0.5 m cube in the glTF importer's quaternion rotation mode: the
# yaw_deg correction has to move it even though rotation_euler stays untouched
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1.0)
cube = bpy.context.active_object
cube.scale = (2.0, 1.0, 0.5)
cube.rotation_mode = "QUATERNION"
bpy.context.view_layer.update()
result["cube"] = assets.normalise_asset(
    [cube], "synthetic", {"synthetic": {"yaw_deg": 90.0}},
    {"api_dimensions_mm": [2000.0, 1000.0, 500.0]},
)
result["cube"]["rotation_mode"] = cube.rotation_mode
print("ORIENTATION " + json.dumps(result))
'''


def run_orientation_probe(binary: str) -> dict:
    """Import the cached coffee table in Blender and return normalise's facts."""
    with tempfile.TemporaryDirectory() as workdir:
        script_path = os.path.join(workdir, "orientation.py")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(ORIENTATION_SCRIPT)
        environment = dict(os.environ, ROOMGEN_REPO_ROOT=REPO_ROOT,
                           COFFEE_GLTF=COFFEE_TABLE_GLTF)
        result = subprocess.run(
            [binary, "--background", "--factory-startup", "--python", script_path],
            capture_output=True, text=True, timeout=600, env=environment,
        )
    if result.returncode != 0:
        raise AssertionError(f"orientation probe failed\n{result.stdout[-3000:]}\n"
                             f"{result.stderr[-3000:]}")
    for line in result.stdout.splitlines():
        if line.startswith("ORIENTATION "):
            return json.loads(line[len("ORIENTATION "):])
    raise AssertionError(f"orientation probe printed no result\n{result.stdout[-3000:]}")


class RegistryAxisAlignmentTest(unittest.TestCase):
    """The loader turns a transposed glTF footprint into the registry frame."""

    API = [1201.8300294876099, 600.0000834465027, 389.99998569488525]

    def test_transposed_footprint_gets_one_quarter_turn(self):
        yaw, note = assets_mod.registry_axis_yaw([0.6, 1.2018, 0.39], self.API)
        self.assertEqual(yaw, 90.0)
        self.assertIn("transpose", note)

    def test_aligned_footprint_is_left_alone(self):
        self.assertEqual(assets_mod.registry_axis_yaw([1.2018, 0.6, 0.39], self.API),
                         (0.0, None))

    def test_square_missing_and_wrong_height_are_not_rotated(self):
        square = [430.0, 431.0, 951.0]
        self.assertEqual(assets_mod.registry_axis_yaw([0.430, 0.431, 0.951], square),
                         (0.0, None))
        self.assertEqual(assets_mod.registry_axis_yaw([0.6, 1.2018, 0.39], None),
                         (0.0, None))
        # a height that disagrees is a real size problem, not an axis-order one
        self.assertEqual(assets_mod.registry_axis_yaw([0.6, 1.2018, 0.50], self.API),
                         (0.0, None))

    def test_manifest_drift_note_clears_for_the_aligned_size(self):
        aligned = COFFEE_TABLE_REGISTRY_M
        self.assertIsNone(checks_mod._built_size_note("coffee_table", aligned, aligned))
        stale = checks_mod._built_size_note("coffee_table", [0.6, 1.2018, 0.39], aligned)
        self.assertIsNotNone(stale)
        self.assertIn("built size differs", stale)


class PolyHavenOrientationTest(unittest.TestCase):
    """Measure the real imported asset: extents and yaw, not just helpers."""

    @classmethod
    def setUpClass(cls):
        binary = blender_binary()
        if binary is None:
            raise unittest.SkipTest("no Blender binary (PANOWORLD_BLENDER / third_party/blender-*)")
        if not os.path.isfile(COFFEE_TABLE_GLTF):
            raise unittest.SkipTest("coffee table glTF is not cached; run assets.py fetch first")
        cls.probe = run_orientation_probe(binary)

    def test_coffee_table_builds_at_the_registry_size(self):
        facts = self.probe["coffee"]
        for measured, registry in zip(facts["size_m"], COFFEE_TABLE_REGISTRY_M):
            self.assertAlmostEqual(measured, registry, places=3)
        self.assertEqual(facts["axis_yaw_deg"], 90.0)
        self.assertEqual(facts["front_yaw_deg"], 0.0)
        self.assertIsNone(facts["scale_warning"])

    def test_yaw_correction_moves_quaternion_mode_geometry(self):
        facts = self.probe["cube"]
        self.assertEqual(facts["rotation_mode"], "QUATERNION")
        self.assertEqual(facts["axis_yaw_deg"], 0.0)
        self.assertEqual(facts["front_yaw_deg"], 90.0)
        self.assertAlmostEqual(facts["size_m"][0], 1.0, places=6)
        self.assertAlmostEqual(facts["size_m"][1], 2.0, places=6)
        self.assertAlmostEqual(facts["size_m"][2], 0.5, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
