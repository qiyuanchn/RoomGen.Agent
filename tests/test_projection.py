"""World point -> camera -> ERP pixel, and the door acceptance built on it.

Run::

    python tests/test_projection.py        # or: python -m pytest tests/test_projection.py

Two things are pinned down here, both of them regressions of the same mistake:

* **the projection** (:func:`tools.validate_blender_agent_p0.world_to_erp` and
  its pure-Python twin :func:`blender_agent.layout_gen.erp_pixel_of_direction`):
  a world point must land on the pixel whose ERP ray points back at it, in all
  four quadrants, with a camera yaw, and across the panorama seam.  The old
  validator located its sample column with a hand-written ``atan2(dy, dx)``,
  which under the ``R0`` convention of ``spec.py`` is the *mirror image* about
  the camera's +X axis: a target to the north-east (1, 1) was sampled to the
  south-east, a target to the north to the south.
* **the door geometry acceptance** (:func:`check_door_openings`) and the
  informational camera coverage (:func:`door_visibility_check`): a real door
  passes, a sealed / wrong-sized / missing-walled door fails, and an occluded
  doorway is reported without failing the scene.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import trimesh  # noqa: E402

from blender_agent import layout_gen  # noqa: E402
from blender_agent import spec as spec_mod  # noqa: E402
import validate_blender_agent_p0 as validator  # noqa: E402

WIDTH, HEIGHT = spec_mod.PANO_WIDTH, spec_mod.PANO_HEIGHT


def _camera_matrix(yaw_deg: float, pos=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array(spec_mod.extrinsics_matrix(list(pos), math.radians(yaw_deg)), dtype=np.float64)


def _ray_grid(c2w: np.ndarray, width: int = WIDTH, height: int = HEIGHT):
    """The validator's ERP ray for every pixel, built once per camera."""
    key = (c2w.tobytes(), width, height)
    if key not in _RAY_CACHE:
        origins, directions = validator.erp_rays(c2w, width, height)
        _RAY_CACHE[key] = (origins, directions)
    return _RAY_CACHE[key]


_RAY_CACHE: dict = {}


def _erp_direction(c2w: np.ndarray, column: int, row: int, width: int = WIDTH, height: int = HEIGHT):
    """The ray of one pixel, straight out of the validator's own ERP builder."""
    _origins, directions = _ray_grid(c2w, width, height)
    return directions[row * width + column]


class ProjectionTest(unittest.TestCase):
    """The mapping has to be the inverse of ``erp_rays``, everywhere."""

    def test_quadrant_columns_are_not_mirrored(self):
        """North-east must go to the north-east column, not to the south-east one."""
        c2w = _camera_matrix(0.0, (0.0, 0.0, 1.5))
        checks = {
            (1.0, 1.0): 0.375,   # north-east: lon = -pi/4
            (1.0, -1.0): 0.625,  # south-east: lon = +pi/4
            (-1.0, -1.0): 0.875,  # south-west: lon = +3pi/4 (wraps)
            (-1.0, 1.0): 0.125,  # north-west: lon = -3pi/4
        }
        for (dx, dy), fraction in checks.items():
            with self.subTest(direction=(dx, dy)):
                u, v, depth = validator.world_to_erp([[dx, dy, 1.5]], c2w, WIDTH, HEIGHT)
                self.assertAlmostEqual(u[0], fraction * WIDTH - 0.5, places=6)
                self.assertAlmostEqual(v[0], HEIGHT / 2.0 - 0.5, places=6)
                self.assertAlmostEqual(depth[0], math.hypot(dx, dy), places=6)

    def test_column_matches_the_ray_that_points_at_the_point(self):
        """The invariant behind the fix: the sampled ray aims at the target."""
        for yaw in (0.0, 37.0, 90.0, -125.0, 180.0):
            c2w = _camera_matrix(yaw, (1.0, -2.0, 1.5))
            for target in ((5.0, 3.0, 1.5), (-4.0, 2.0, 0.4), (2.0, -6.0, 2.4), (-3.0, -3.0, 1.0)):
                with self.subTest(yaw=yaw, target=target):
                    u, v, distance = validator.world_to_erp([target], c2w, WIDTH, HEIGHT)
                    self.assertLessEqual(abs(u[0] - round(u[0])), 0.5 + 1e-9)
                    self.assertLessEqual(abs(v[0] - round(v[0])), 0.5 + 1e-9)
                    # ... and that pixel's ray points at it (within the pixel size)
                    column, row, _ = validator.erp_pixel_of_point(target, c2w, WIDTH, HEIGHT)
                    ray = _erp_direction(c2w, column, row)
                    offset = np.array(target) - c2w[:3, 3]
                    unit = offset / np.linalg.norm(offset)
                    self.assertGreater(float(np.dot(ray, unit)), 1.0 - 1e-5,
                                       f"the ray of column {column} does not aim at {target}")
                    self.assertAlmostEqual(float(distance[0]), float(np.linalg.norm(offset)), places=6)

    def test_the_old_atan2_column_would_have_been_a_different_pixel(self):
        """Regression for the review: ``atan2(dy, dx)`` is the mirror column."""
        c2w = _camera_matrix(0.0)
        for target in ((1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)):
            dx, dy = target[0], target[1]
            old = int(round((math.atan2(dy, dx) + math.pi) / (2.0 * math.pi) * WIDTH - 0.5)) % WIDTH
            column, _row, _distance = validator.erp_pixel_of_point(target, c2w, WIDTH, HEIGHT)
            with self.subTest(target=target):
                self.assertNotEqual(old, column)
                mirror = int(round((math.atan2(-dy, dx) + math.pi) / (2.0 * math.pi) * WIDTH - 0.5)) % WIDTH
                # the mirror is the *correct* answer, and it is what the projection gives
                self.assertLessEqual(abs(((column - mirror + WIDTH // 2) % WIDTH) - WIDTH // 2), 1)

    def test_pixel_ray_round_trip_including_the_seam(self):
        """Every sampled pixel, seam columns included, projects back to itself."""
        for yaw in (0.0, 61.0):
            c2w = _camera_matrix(yaw, (0.5, 0.25, 1.5))
            columns = (0, 1, 2, WIDTH // 4, WIDTH // 2, WIDTH - 3, WIDTH - 2, WIDTH - 1)
            rows = (0, 1, HEIGHT // 2, HEIGHT - 2, HEIGHT - 1)
            for column in columns:
                for row in rows:
                    ray = _erp_direction(c2w, column, row)
                    point = c2w[:3, 3] + ray * 6.0
                    u, v, _depth = validator.world_to_erp([point], c2w, WIDTH, HEIGHT)
                    wrapped = (u[0] - column + WIDTH / 2.0) % WIDTH - WIDTH / 2.0
                    with self.subTest(yaw=yaw, column=column, row=row):
                        self.assertLess(abs(wrapped), 1e-4,
                                        f"column {column} projects back to {u[0]} (seam wrap)")
                        self.assertAlmostEqual(v[0], row, places=4)

    def test_layout_gen_projection_matches_the_validator(self):
        """The generator's forecast and the validator measure the same pixel."""
        rng = np.random.default_rng(7)
        for yaw in (0.0, 90.0, -30.0):
            c2w = _camera_matrix(yaw, (0.3, -0.4, 1.5))
            rotation = c2w[:3, :3].tolist()
            targets = np.column_stack([
                rng.uniform(-8.0, 8.0, 64),
                rng.uniform(-8.0, 8.0, 64),
                rng.uniform(0.1, 2.6, 64),
            ])
            for target in targets:
                column, row, _distance = validator.erp_pixel_of_point(target, c2w, WIDTH, HEIGHT)
                other = layout_gen.erp_pixel_of_direction(np.array(target) - c2w[:3, 3], rotation)
                with self.subTest(yaw=yaw, target=target):
                    self.assertEqual((column, row), other)


class DoorGeometryTest(unittest.TestCase):
    """Per-opening acceptance: real door passes, sealed / wrong door fails."""

    SPEC = {
        "unit": "meter",
        "wall_height": 2.7,
        "wall_thickness": 0.12,
        "rooms": [
            {"id": "a", "type": "room", "polygon": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]},
            {"id": "b", "type": "room", "polygon": [[4.0, 0.0], [8.0, 0.0], [8.0, 3.0], [4.0, 3.0]]},
        ],
        "openings": [{"kind": "door", "between": ["a", "b"], "center": [4.0, 1.5],
                      "width": 0.9, "height": 2.1}],
        "cameras": [{"name": "0000", "room": "a", "pos": [2.0, 1.5, 1.5]}],
    }
    WALL_HALF = 0.06

    @classmethod
    def setUpClass(cls):
        cls.spec = spec_mod.parse_spec(dict(cls.SPEC))

    def _caster(self, opening: tuple, wall_height: float = 2.7):
        """A wall at x=4 with a hole ``(y0, y1, z_top)`` cut out of it."""
        y0, y1, z_top = opening
        boxes = []
        if y0 > 0.0:
            boxes.append(self._box(4.0 - self.WALL_HALF, 0.0, 4.0 + self.WALL_HALF, y0, 0.0, wall_height))
        if y1 < 3.0:
            boxes.append(self._box(4.0 - self.WALL_HALF, y1, 4.0 + self.WALL_HALF, 3.0, 0.0, wall_height))
        if z_top < wall_height:
            boxes.append(self._box(4.0 - self.WALL_HALF, y0, 4.0 + self.WALL_HALF, y1, z_top, wall_height))
        mesh = trimesh.util.concatenate(boxes)
        return validator.MeshRayCaster(mesh)

    @staticmethod
    def _box(x0, y0, x1, y1, z0, z1) -> trimesh.Trimesh:
        return trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0)).apply_translation(
            ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0))

    def test_a_real_door_passes_and_is_measured(self):
        report = validator.check_door_openings(self.spec, self._caster((1.05, 1.95, 2.1)))
        self.assertTrue(report["pass"], report["doors"][0]["failures"])
        door = report["doors"][0]
        self.assertAlmostEqual(door["measured_width_m"], 0.90, delta=0.04)
        self.assertAlmostEqual(door["measured_height_m"], 2.10, delta=0.04)
        for room_id in ("a", "b"):
            side = door["sides"][room_id]
            self.assertGreaterEqual(side["open_fraction"], 0.98)
            self.assertGreaterEqual(side["wall_solid_fraction"], 0.95)

    def test_sealed_door_fails(self):
        """A wall where the spec declares a door must fail, whatever the camera does."""
        report = validator.check_door_openings(self.spec, self._caster((1.5, 1.5, 2.7)))
        self.assertFalse(report["pass"])
        self.assertTrue(any("does not reach the floor" in failure or "not open" in failure
                            for failure in report["doors"][0]["failures"]),
                        report["doors"][0]["failures"])

    def test_furniture_in_the_doorway_does_not_change_the_wall_contract(self):
        """The geometry check measures the *shell*: a blocked room is not a sealed door."""
        spec = spec_mod.parse_spec({
            **self.SPEC,
            "furniture": [{"asset": "ph:sofa_02", "room": "b", "pos": [4.6, 1.5], "rot_z_deg": 90}],
        })
        report = validator.check_door_openings(spec, self._caster((1.05, 1.95, 2.1)))
        self.assertTrue(report["pass"], report["doors"][0]["failures"])

    def test_wrong_opening_width_fails(self):
        report = validator.check_door_openings(self.spec, self._caster((0.8, 2.2, 2.1)))
        self.assertFalse(report["pass"])
        self.assertTrue(any("measured opening width" in failure for failure in report["doors"][0]["failures"]),
                        report["doors"][0]["failures"])

    def test_wrong_opening_height_fails(self):
        report = validator.check_door_openings(self.spec, self._caster((1.05, 1.95, 1.7)))
        self.assertFalse(report["pass"])
        self.assertTrue(any("measured opening height" in failure for failure in report["doors"][0]["failures"]),
                        report["doors"][0]["failures"])

    def test_missing_wall_beside_the_door_fails(self):
        """A "door" that is really a missing wall is not an opening either."""
        report = validator.check_door_openings(self.spec, self._caster((0.0, 3.0, 2.1)))
        self.assertFalse(report["pass"])
        self.assertTrue(any("wall is missing" in failure for failure in report["doors"][0]["failures"]),
                        report["doors"][0]["failures"])

    def test_every_opening_is_checked_not_just_the_first(self):
        """Two doors: sealing the *second* one has to fail."""
        spec = spec_mod.parse_spec({
            **self.SPEC,
            "rooms": [
                {"id": "a", "type": "room", "polygon": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]},
                {"id": "b", "type": "room", "polygon": [[4.0, 0.0], [8.0, 0.0], [8.0, 3.0], [4.0, 3.0]]},
                {"id": "c", "type": "room", "polygon": [[8.0, 0.0], [12.0, 0.0], [12.0, 3.0], [8.0, 3.0]]},
            ],
            "openings": [
                {"kind": "door", "between": ["a", "b"], "center": [4.0, 1.5], "width": 0.9, "height": 2.1},
                {"kind": "door", "between": ["b", "c"], "center": [8.0, 1.5], "width": 0.9, "height": 2.1},
            ],
        })
        open_wall = self._box(3.94, 0.0, 4.06, 1.05, 0.0, 2.7)
        open_jamb = self._box(3.94, 1.95, 4.06, 3.0, 0.0, 2.7)
        open_lintel = self._box(3.94, 1.05, 4.06, 1.95, 2.1, 2.7)
        sealed = self._box(7.94, 0.0, 8.06, 3.0, 0.0, 2.7)
        caster = validator.MeshRayCaster(trimesh.util.concatenate(
            [open_wall, open_jamb, open_lintel, sealed]))
        report = validator.check_door_openings(spec, caster)
        self.assertFalse(report["pass"])
        self.assertEqual(len(report["doors"]), 2)
        self.assertTrue(report["doors"][0]["pass"])
        self.assertFalse(report["doors"][1]["pass"], report["doors"][1])


class DoorVisibilityMetricTest(unittest.TestCase):
    """The camera-side number: informational, and not fooled by the mirror bug."""

    SPEC = {
        "unit": "meter",
        "wall_height": 2.7,
        "wall_thickness": 0.12,
        "rooms": [
            {"id": "a", "type": "room", "polygon": [[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]]},
            {"id": "b", "type": "room", "polygon": [[6.0, 0.0], [10.0, 0.0], [10.0, 6.0], [6.0, 6.0]]},
        ],
        "openings": [{"kind": "door", "between": ["a", "b"], "center": [6.0, 4.0],
                      "width": 0.9, "height": 2.1}],
        "cameras": [{"name": "0000", "room": "a", "pos": [1.0, 1.0, 1.5]}],
    }

    def setUp(self):
        self.spec = spec_mod.parse_spec(dict(self.SPEC))
        self.camera = self.spec.cameras[0]
        self.c2w = _camera_matrix(0.0, self.camera.pos)

    def _rendered(self, door_pixel_value: float, elsewhere: float, mirrored_value: float = None):
        """A fake depth image: the door pixel reads ``door_pixel_value``."""
        point = [6.0, 4.0, 1.05]
        column, row, _distance = validator.erp_pixel_of_point(point, self.c2w, WIDTH, HEIGHT)
        image = np.full((HEIGHT, WIDTH), elsewhere)
        image[max(0, row - 2):row + 3, max(0, column - 2):column + 3] = door_pixel_value
        if mirrored_value is not None:
            # the column the old atan2(dy, dx) formula would have sampled
            to_door = (6.0 - 1.0, 4.0 - 1.0)
            old_column = int(round((math.atan2(to_door[1], to_door[0]) + math.pi)
                                   / (2.0 * math.pi) * WIDTH - 0.5)) % WIDTH
            image[max(0, row - 2):row + 3, max(0, old_column - 2):old_column + 3] = mirrored_value
        return image

    def test_the_door_pixel_is_the_projected_one_not_the_mirrored_one(self):
        """Old code sampled the mirror column: a wall there must not hide the door."""
        rendered = self._rendered(door_pixel_value=9.0, elsewhere=3.0, mirrored_value=0.5)
        report = validator.door_visibility_check(self.spec, self.camera, rendered, self.c2w)
        self.assertEqual(report["visible"], 1)
        self.assertTrue(report["doors"][0]["visible"])
        self.assertTrue(report["pass"])
        self.assertEqual(report["role"], "informational")

    def test_occlusion_is_reported_but_does_not_fail_the_scene(self):
        """A camera that cannot see its doorway is a number, not a verdict."""
        rendered = np.full((HEIGHT, WIDTH), 1.2)
        report = validator.door_visibility_check(self.spec, self.camera, rendered, self.c2w)
        self.assertEqual(report["visible"], 0)
        self.assertGreater(report["of"], 0)
        self.assertTrue(report["pass"], "reasonable occlusion must not fail the scene")

    def test_a_camera_with_no_door_reports_nothing(self):
        spec = spec_mod.parse_spec({**self.SPEC,
                                    "openings": []})
        camera = spec.cameras[0]
        report = validator.door_visibility_check(spec, camera, np.zeros((HEIGHT, WIDTH)), self.c2w)
        self.assertTrue(report["pass"])
        self.assertIn("skipped", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
