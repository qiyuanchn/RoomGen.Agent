"""Unit tests for the physics settling stage (host python, no bpy).

Run::

    python tests/test_settle.py               # or: python tests/test_settle.py -v

Covers the parts of ``blender_agent/settle.py`` that do not need Blender: the
configuration resolution (spec block / environment / caller), the rest-vs-held
classification (checked against the shipped P1 spec), the acceptance maths, and
the geometric helpers (base flattening, minimum translation).  When a Blender
binary is available the physics itself is verified by driving the module's
headless self-check over placed boxes (floating, sunk, overlapping, stacked, on
a rug, in a wall, instanced) -- see ``settle.selfcheck``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import settle  # noqa: E402
from blender_agent import spec as spec_mod  # noqa: E402

P1_SPEC = os.path.join(REPO_ROOT, "configs", "blender_agent_p1_furnished.json")

#: the P1 spec dresses the rooms with pieces that stand on the floor and with
#: pieces that hang on a wall / ceiling or are stacked on furniture
P1_RESTING = {"sofa", "coffee_table", "rug", "armchair", "bookshelf", "plant_tall",
              "plant_corner", "basket", "bed", "nightstand", "plant_bedroom"}
P1_HELD = {"books_a", "books_b", "books_c", "books_d", "books_e", "ceiling_lamp",
           "art_sofa", "art_west", "bedside_lamp", "ceiling_lamp_bed", "art_bed"}


def _have_numpy() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def blender_binary():
    """The Blender used by the runner scripts, or ``None`` when not installed."""
    candidate = os.environ.get("PANOWORLD_BLENDER") or os.path.join(
        REPO_ROOT, "third_party", "blender-4.5.14-linux-x64", "blender")
    return candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else None


def _placements_from(spec):
    """The placement records ``run_build.place_furniture`` would write (names/z/mount)."""
    placements = []
    for index, item in enumerate(spec.furniture):
        name = item.name or f"item_{index}"
        height = float(item.z) if item.z is not None else 0.0
        if item.mount == "ceiling":
            height = spec.wall_height - 0.5
        record = {"name": name, "asset": item.asset, "kind": "builtin" if item.asset.startswith("builtin:")
                  else "model", "pos": list(item.pos), "rot_z_deg": item.rot_z_deg, "z": height,
                  "mount": item.mount}
        if record["kind"] == "model":
            record["instance"] = f"furn_{name}"
            record["bbox_z"] = [height, height + 0.5]
        placements.append(record)
    return placements


class ModuleContractTest(unittest.TestCase):
    def test_host_import_does_not_pull_in_bpy(self):
        self.assertNotIn("bpy", sys.modules, "settle.py must import bpy lazily")

    def test_spec_block_is_not_part_of_the_schema(self):
        # the module reads a top-level "settle" block; spec.py must ignore it
        spec = spec_mod.load_spec(P1_SPEC)
        self.assertNotIn(settle.SPEC_BLOCK, spec_mod.dump_manifest_spec(spec))


class ConfigTest(unittest.TestCase):
    def test_defaults_without_a_spec(self):
        config = settle.resolve_config(None)
        self.assertFalse(config["enabled"])
        self.assertEqual(config["margin_m"], settle.DEFAULT_MARGIN_M)
        self.assertEqual(config["drop_m"], settle.DEFAULT_DROP_M)
        self.assertGreater(config["ground_tolerance_m"], config["margin_m"] * 2)

    def test_auto_enables_for_a_furnished_spec(self):
        spec = spec_mod.load_spec(P1_SPEC)
        config = settle.resolve_config(spec)
        self.assertTrue(config["enabled"])
        self.assertIn("auto", config["reason"])

    def test_auto_skips_a_front3d_replay(self):
        spec = spec_mod.load_spec(P1_SPEC)
        spec.front3d = {"scene_json": "/data/3D-FRONT/x.json"}
        config = settle.resolve_config(spec)
        self.assertFalse(config["enabled"])
        self.assertIn("front3d", config["reason"])
        # ... unless it is forced
        self.assertTrue(settle.resolve_config(spec, environ={"PANOWORLD_SETTLE": "1"})["enabled"])

    def test_environment_and_caller_override_the_spec_block(self):
        spec = spec_mod.load_spec(P1_SPEC)
        spec.raw[settle.SPEC_BLOCK] = {"enabled": False, "margin_m": 0.01, "drop_m": 0.05,
                                       "substeps_per_frame": 8}
        config = settle.resolve_config(spec, environ={"PANOWORLD_SETTLE_MARGIN": "0.004"})
        self.assertFalse(config["enabled"])          # spec block wins over auto
        self.assertEqual(config["margin_m"], 0.004)  # env wins over the spec block
        self.assertEqual(config["drop_m"], 0.05)
        self.assertEqual(config["substeps"], 8)      # alias
        forced = settle.resolve_config(spec, environ={"PANOWORLD_SETTLE": "on", "PANOWORLD_SETTLE_FRAMES": "12"})
        self.assertTrue(forced["enabled"])
        self.assertEqual(forced["frames"], 12)
        overrides = settle.resolve_config(spec, environ={"PANOWORLD_SETTLE": "1"}, frames=99, angular_damping=0.5)
        self.assertEqual(overrides["frames"], 99)    # caller wins over env
        self.assertEqual(overrides["angular_damping"], 0.5)

    def test_bad_values_are_rejected(self):
        with self.assertRaises(ValueError):
            settle.resolve_config(None, environ={"PANOWORLD_SETTLE": "maybe"})
        with self.assertRaises(ValueError):
            settle.resolve_config(None, margin_m=-0.01)
        with self.assertRaises(ValueError):
            settle.resolve_config(None, frames=0)

    def test_notes_record_the_source_of_every_value(self):
        config = settle.resolve_config(None, environ={"PANOWORLD_SETTLE_DROP": "0.03"})
        self.assertEqual(config["drop_m"], 0.03)
        self.assertTrue(any("drop_m=0.03" in note and "environment" in note for note in config["notes"]))


class ClassificationTest(unittest.TestCase):
    def setUp(self):
        self.spec = spec_mod.load_spec(P1_SPEC)
        self.placements = _placements_from(self.spec)
        self.config = settle.resolve_config(self.spec)

    def test_p1_spec_splits_into_resting_and_held(self):
        decisions = settle.plan(self.spec, self.placements, self.config)
        self.assertEqual(len(decisions), len(self.spec.furniture))
        resting = {d["name"] for d in decisions if d["mode"] == "rest"}
        held = {d["name"] for d in decisions if d["mode"] == "held"}
        self.assertEqual(resting, P1_RESTING)
        self.assertEqual(held, P1_HELD)

    def test_reasons_name_the_mount_rule(self):
        for decision in settle.plan(self.spec, self.placements, self.config):
            if decision["mode"] == "held":
                self.assertTrue("z=" in decision["reason"] or "mount" in decision["reason"],
                                f"{decision['name']}: {decision['reason']}")

    def test_mount_rules_and_lift_tolerance(self):
        config = settle.resolve_config(None)
        on_floor = {"z": 0.0, "mount": "floor"}
        self.assertEqual(settle.placement_mode(None, on_floor, config)[0], "rest")
        self.assertEqual(settle.placement_mode(None, {"z": 0.015}, config)[0], "rest")
        self.assertEqual(settle.placement_mode(None, {"z": 0.52}, config)[0], "held")
        self.assertEqual(settle.placement_mode(None, {"z": 0.0, "mount": "ceiling"}, config)[0], "held")
        self.assertEqual(settle.placement_mode(None, {"z": 1.2, "mount": "wall"}, config)[0], "held")
        # a placement the spec does not describe still gets a decision
        self.assertEqual(settle.placement_mode(None, {"z": None}, config)[0], "rest")


class AcceptanceMathsTest(unittest.TestCase):
    def setUp(self):
        self.config = settle.resolve_config(None)

    def test_ground_check(self):
        resting = settle.ground_check(0.002, self.config)
        self.assertTrue(resting["grounded"])
        self.assertFalse(resting["penetrating"])
        self.assertEqual(resting["gap_m"], 0.002)
        sunk = settle.ground_check(-0.004, self.config)
        self.assertTrue(sunk["penetrating"])
        floating = settle.ground_check(0.031, self.config)
        self.assertFalse(floating["grounded"])
        self.assertAlmostEqual(floating["floating_m"], 0.011)

    def test_bbox_z(self):
        if not _have_numpy():
            self.skipTest("numpy is not installed for this interpreter")
        import numpy as np
        points = np.array([[0.0, 0.0, 0.1], [1.0, 2.0, 0.9], [0.5, 0.5, 0.4]])
        self.assertEqual(settle.bbox_z(points), [0.1, 0.9])
        self.assertIsNone(settle.bbox_z(np.zeros((0, 3))))


class GeometryHelperTest(unittest.TestCase):
    def setUp(self):
        if not _have_numpy():
            self.skipTest("numpy is not installed for this interpreter")

    def test_flatten_base_projects_the_footprint_onto_the_lowest_plane(self):
        import numpy as np
        # a pot whose base is a rough disc: the lowest vertex is a lone spike
        ring = np.array([[np.cos(a) * 0.13, np.sin(a) * 0.13, 0.004] for a in np.linspace(0, 2 * np.pi, 12, False)])
        spike = np.array([[0.0, 0.0, 0.0]])
        canopy = np.array([[np.cos(a) * 0.3, np.sin(a) * 0.3, 0.8] for a in np.linspace(0, 2 * np.pi, 12, False)])
        points = np.vstack([ring, spike, canopy])
        flattened = settle.flatten_base(points, 0.015)
        self.assertEqual(flattened[:, 2].min(), 0.0)                  # the height does not change
        bottom = flattened[flattened[:, 2] < 1e-9]
        self.assertGreaterEqual(len(bottom), 13)                      # spike + the whole ring
        self.assertGreater(np.ptp(bottom[:, 0]), 0.24)                # a real patch, not a point
        self.assertEqual(len(settle.flatten_base(points, 0.0)), len(points))

    def test_minimum_translation_separates_with_the_smallest_motion(self):
        import numpy as np
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        b = a + np.array([0.8, 0.0, 0.0])     # 0.2 m of overlap along x
        directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        direction, distance = settle.minimum_translation(a, b, directions, eps=0.002)
        self.assertGreater(distance, 0.2)
        self.assertLess(distance, 0.21)
        np.testing.assert_allclose(np.abs(direction), [1.0, 0.0, 0.0], atol=1e-9)
        moved = b - direction * 0.2
        self.assertGreaterEqual(float((a[:, 0].max() - moved[:, 0].min()) * direction[0]), -1e-9)
        # already separated along x -> no motion proposed for that axis
        far = a + np.array([3.0, 0.0, 0.0])
        direction, distance = settle.minimum_translation(a, far, directions, eps=0.002)
        self.assertIsNone(direction)
        self.assertEqual(distance, 0.0)


class _FakeObject:
    def __init__(self, name, type="MESH"):
        self.name = name
        self.type = type


class TargetMatchingTest(unittest.TestCase):
    """Generated builtin props are grouped by creation order, not by name."""

    def test_duplicate_names_get_distinct_roots(self):
        self.assertEqual(settle._prop_root("rug_living"), "rug_living")
        self.assertEqual(settle._prop_root("rug_living.001"), "rug_living.001")
        self.assertEqual(settle._prop_root("lamp_bed_base.001"), "lamp_bed.001")
        self.assertEqual(settle._prop_root("lamp_bed_stem.001"), "lamp_bed.001")
        self.assertEqual(settle._prop_root("lamp_bed_shade.001"), "lamp_bed.001")
        self.assertEqual(settle._strip_duplicate_suffix("rug_living.001"), "rug_living")

    def test_generated_groups_follow_creation_order(self):
        objects = [
            _FakeObject("rug_living"),
            _FakeObject("lamp_bed_base"), _FakeObject("lamp_bed_stem"), _FakeObject("lamp_bed_shade"),
            _FakeObject("cabinet_carcass"),          # a non-prop mesh breaks the run
            _FakeObject("rug_living.001"),
            _FakeObject("lamp_bed_base.001"), _FakeObject("lamp_bed_stem.001"),
            _FakeObject("lamp_bed_shade.001"),
        ]
        groups = settle._generated_prop_groups(objects)
        self.assertEqual(
            [[obj.name for obj in group] for group in groups],
            [["rug_living"],
             ["lamp_bed_base", "lamp_bed_stem", "lamp_bed_shade"],
             ["rug_living.001"],
             ["lamp_bed_base.001", "lamp_bed_stem.001", "lamp_bed_shade.001"]],
        )


class PlacementBookkeepingTest(unittest.TestCase):
    """Settle writes its results back by placement index, not by name."""

    def _entry(self, index, name, objects, bottom=0.002):
        return {
            "index": index, "name": name, "mode": "rest", "reason": "rests on the floor",
            "objects": objects, "bbox_z_before": [0.02, 0.82],
            "bbox_z_after": [bottom, 0.80], "moved_m": 0.018, "rotation_deg": 0.0,
            "lift_m": 0.021, "separation_m": 0.0, "flat": False,
            "gap_m": 0.002, "grounded": True, "penetrating": False,
        }

    def test_duplicate_names_are_written_to_their_own_placement(self):
        placements = [
            {"name": "kitchen_cabinet", "z": 0.0, "bbox_z": [0.02, 0.82]},
            {"name": "kitchen_cabinet", "z": 0.0, "bbox_z": [0.02, 0.82]},
        ]
        # deliberately out of order: only the index may pair them
        contents = [self._entry(1, "kitchen_cabinet", ["cabinet_b"], bottom=0.003),
                    self._entry(0, "kitchen_cabinet", ["cabinet_a"], bottom=0.002)]
        settle._update_placements(placements, contents, settle.resolve_config(None))
        self.assertEqual(placements[0]["settle"]["objects"], ["cabinet_a"])
        self.assertEqual(placements[1]["settle"]["objects"], ["cabinet_b"])
        self.assertEqual(placements[0]["z"], 0.002)
        self.assertEqual(placements[1]["z"], 0.003)
        self.assertEqual(placements[0]["settle"]["bbox_z_after"], [0.002, 0.80])


class HeadlessSelfCheckTest(unittest.TestCase):
    """Drive the CLI self-check: placed boxes must end up on the floor, unsunk."""

    def test_selfcheck_passes(self):
        binary = blender_binary()
        if binary is None:
            self.skipTest("no Blender binary (PANOWORLD_BLENDER / third_party/blender-*)")
        with tempfile.TemporaryDirectory() as workdir:
            script = os.path.join(REPO_ROOT, "blender_agent", "settle.py")
            result = subprocess.run(
                [binary, "--background", "--factory-startup", "--python", script,
                 "--", "--selfcheck", "--json", workdir],
                capture_output=True, text=True, timeout=600,
            )
            report_path = os.path.join(workdir, "settle_selfcheck.json")
            self.assertTrue(os.path.isfile(report_path),
                            f"self-check wrote no report\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        self.assertEqual(result.returncode, 0, result.stdout[-2000:])
        self.assertTrue(report["pass"], json.dumps(report["checks"], indent=2))
        settle_report = report["report"]
        self.assertTrue(settle_report["checks"]["pass"], settle_report["checks"]["violations"])
        self.assertTrue(settle_report["simulation"]["at_rest"])
        self.assertLessEqual(settle_report["checks"]["max_bottom_gap_m"],
                             settle_report["checks"]["ground_tolerance_m"])
        self.assertGreaterEqual(settle_report["checks"]["min_bottom_m"], -0.001)
        # a failed check must fail the process (the runner scripts rely on the exit code)
        self.assertEqual({item["pass"] for item in report["checks"]}, {True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
