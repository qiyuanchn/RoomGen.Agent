"""Tests for the Blender perspective RGB-D observation backend.

Run::

    /opt/conda/envs/diffaero/bin/python tests/test_observation_blender.py

The unit half pins the coordinate / intrinsics contract without Blender.  The
integration half builds a small room with *known* geometry (interior walls at
x = +-3 m, y = +-1.5 m, z = +-2 m with distinct colours) and renders it with
headless Blender:

* RGB is non-empty and castable to ``uint8``;
* the centre depth equals the geometric wall distance to within 1 cm;
* yaw 0 (Habitat) sees the red -Z wall and yaw +90 the green -X wall;
* pitch -45 / +45 hit the floor / ceiling at 1.5 / cos(45 deg);
* Euclidean ``distance`` and z-depth agree pixel by pixel;
* reset/final capture writes loadable PNG/NPY files.

If Blender is not installed the integration tests skip; everything else runs
on a normal host Python.
"""

from __future__ import annotations

import json
import math
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

from roomgen_nav import observation_blender as ob  # noqa: E402

try:
    import trimesh
    from trimesh.visual.material import PBRMaterial
except ImportError:  # pragma: no cover - trimesh is in requirements.txt
    trimesh = None
    PBRMaterial = None


# --------------------------------------------------------------------------- #
# known-geometry fixture
# --------------------------------------------------------------------------- #
ROOM_HALF_X = 3.0
ROOM_HALF_Y = 1.5
ROOM_HALF_Z = 2.0


def _coloured_slab(extents, translation, colour, name):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(translation)
    mesh.invert()  # the camera sits inside: face normals point inwards
    mesh.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
        name=name,
        baseColorFactor=[colour[0], colour[1], colour[2], 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.9,
    ))
    return mesh


MARKER_X = 1.0
MARKER_Z = ROOM_HALF_Z - 0.03


def build_known_room(path: str) -> str:
    """Six inward-facing slabs plus a small green marker on the red -Z wall.

    The marker (a 0.2 m square at x = +1 m) is what proves the horizontal
    projection: at yaw 0 it must land at ``u = width / 2 + fx * x / |z|`` on the
    right half of the image, not mirrored onto the left.
    """
    thickness = 0.1
    marker = trimesh.creation.box(extents=(0.2, 0.2, 0.02))
    marker.apply_translation((MARKER_X, 0.0, -MARKER_Z))
    marker.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
        name="green_marker", baseColorFactor=[0.04, 0.70, 0.04, 1.0],
        metallicFactor=0.0, roughnessFactor=0.9))
    meshes = [
        _coloured_slab((2 * ROOM_HALF_X + 0.2, thickness, 2 * ROOM_HALF_Z + 0.2),
                       (0, -ROOM_HALF_Y - thickness / 2, 0), (0.85, 0.85, 0.82), "floor"),
        _coloured_slab((2 * ROOM_HALF_X + 0.2, thickness, 2 * ROOM_HALF_Z + 0.2),
                       (0, ROOM_HALF_Y + thickness / 2, 0), (0.85, 0.85, 0.82), "ceiling"),
        _coloured_slab((2 * ROOM_HALF_X + 0.2, 2 * ROOM_HALF_Y + 0.2, thickness),
                       (0, 0, -ROOM_HALF_Z - thickness / 2), (0.80, 0.04, 0.04), "red_-Z"),
        _coloured_slab((2 * ROOM_HALF_X + 0.2, 2 * ROOM_HALF_Y + 0.2, thickness),
                       (0, 0, ROOM_HALF_Z + thickness / 2), (0.10, 0.15, 0.70), "blue_+Z"),
        _coloured_slab((thickness, 2 * ROOM_HALF_Y + 0.2, 2 * ROOM_HALF_Z + 0.2),
                       (-ROOM_HALF_X - thickness / 2, 0, 0), (0.04, 0.70, 0.04), "green_-X"),
        _coloured_slab((thickness, 2 * ROOM_HALF_Y + 0.2, 2 * ROOM_HALF_Z + 0.2),
                       (ROOM_HALF_X + thickness / 2, 0, 0), (0.55, 0.55, 0.55), "grey_+X"),
        marker,
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trimesh.Scene(meshes).export(path)
    return path


# --------------------------------------------------------------------------- #
# pure-Python contract
# --------------------------------------------------------------------------- #
class CoordinateContractTest(unittest.TestCase):
    def test_gltf_blender_roundtrip(self):
        point = (1.25, -2.5, 3.75)
        np.testing.assert_allclose(ob.blender_to_gltf(ob.gltf_to_blender(point)), point)
        np.testing.assert_allclose(ob.gltf_to_blender((1.0, 2.0, 3.0)), (1.0, -3.0, 2.0))

    def test_habitat_yaw_axes(self):
        forward, right, up = ob.camera_basis(0.0, 0.0, "habitat")
        np.testing.assert_allclose(forward, (0.0, 0.0, -1.0), atol=1e-12)
        np.testing.assert_allclose(right, (1.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(up, (0.0, 1.0, 0.0), atol=1e-12)
        forward, _right, _up = ob.camera_basis(90.0, 0.0, "habitat")
        np.testing.assert_allclose(forward, (-1.0, 0.0, 0.0), atol=1e-12)
        # +90 about +Y is a right-handed rotation; -90 looks along +X.
        forward, _right, _up = ob.camera_basis(-90.0, 0.0, "habitat")
        np.testing.assert_allclose(forward, (1.0, 0.0, 0.0), atol=1e-12)

    def test_manifest_yaw_matches_exporter(self):
        # blender_agent.export_glb.blender_heading_to_gltf: yaw 0 -> +X, 90 -> -Z.
        forward, _right, _up = ob.camera_basis(0.0, 0.0, "manifest")
        np.testing.assert_allclose(forward, (1.0, 0.0, 0.0), atol=1e-12)
        forward, _right, _up = ob.camera_basis(90.0, 0.0, "manifest")
        np.testing.assert_allclose(forward, (0.0, 0.0, -1.0), atol=1e-12)

    def test_pitch_positive_is_up(self):
        forward, _right, _up = ob.camera_basis(0.0, 45.0, "habitat")
        self.assertAlmostEqual(float(forward[1]), math.sin(math.radians(45.0)), places=12)
        self.assertLess(float(forward[2]), 0.0)

    def test_intrinsics_pinhole(self):
        intrinsics = ob.pinhole_intrinsics(128, 128, 90.0)
        np.testing.assert_allclose(intrinsics, [[64.0, 0.0, 64.0],
                                                [0.0, 64.0, 64.0],
                                                [0.0, 0.0, 1.0]])
        wide = ob.pinhole_intrinsics(640, 480, 90.0)
        self.assertAlmostEqual(float(wide[0, 0]), 320.0)
        self.assertAlmostEqual(float(wide[1, 1]), 320.0)  # square pixels
        self.assertAlmostEqual(float(wide[0, 2]), 320.0)
        self.assertAlmostEqual(float(wide[1, 2]), 240.0)

    def test_extrinsics_are_camera_to_world(self):
        position = (2.0, 1.5, -3.0)
        extrinsic = ob.camera_to_world(position, 0.0, 0.0, "habitat")
        np.testing.assert_allclose(extrinsic[:3, 3], position)
        rotation = extrinsic[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)
        # Camera looks along -Z; column 2 is the backward (+Z) axis.
        np.testing.assert_allclose(-rotation[:, 2], (0.0, 0.0, -1.0), atol=1e-12)

    def test_bad_convention(self):
        with self.assertRaises(ValueError):
            ob.camera_basis(0.0, 0.0, "opengl")
        with self.assertRaises(ValueError):
            ob.pinhole_intrinsics(0, 128, 90.0)
        with self.assertRaises(ValueError):
            ob.pinhole_intrinsics(128, 128, 190.0)

    def test_scene_asset_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            direct = os.path.join(tmp, "custom.glb")
            open(direct, "wb").close()
            self.assertEqual(ob.find_scene_asset(direct), direct)
            self.assertEqual(ob.find_scene_asset(tmp), direct)  # only top-level glb

            nested = os.path.join(tmp, "house")
            os.makedirs(os.path.join(nested, "build"))
            build_glb = os.path.join(nested, "build", "scene.glb")
            open(build_glb, "wb").close()
            self.assertEqual(ob.find_scene_asset(nested), build_glb)

            with self.assertRaises(FileNotFoundError):
                ob.find_scene_asset(os.path.join(tmp, "missing"))
            bad = os.path.join(tmp, "notes.txt")
            open(bad, "w").close()
            with self.assertRaises(ValueError):
                ob.find_scene_asset(bad)

    def test_observation_arrays_and_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            observation = ob.Observation(
                rgb=np.full((4, 6, 3), 7, dtype=np.uint8),
                depth=np.full((4, 6), 2.5, dtype=np.float32),
                distance=np.full((4, 6), 2.6, dtype=np.float32),
                position=(1.0, 1.5, 2.0), yaw=10.0, pitch=5.0,
                yaw_convention="habitat", resolution=(6, 4),
                intrinsics=ob.pinhole_intrinsics(6, 4, 90.0),
                extrinsics=ob.camera_to_world((1.0, 1.5, 2.0), 10.0, 5.0),
                metadata={"test": True},
            )
            self.assertEqual(observation.rgb.shape, (4, 6, 3))
            self.assertEqual(observation.rgb.dtype, np.uint8)
            self.assertEqual(observation.depth.dtype, np.float32)
            paths = observation.save(os.path.join(tmp, "frame"))
            for path in paths.values():
                self.assertTrue(os.path.isfile(path), path)
            from PIL import Image
            with Image.open(paths["rgb"]) as image:
                np.testing.assert_array_equal(np.asarray(image.convert("RGB")), observation.rgb)
            np.testing.assert_allclose(np.load(paths["depth"]), observation.depth)
            with open(paths["meta"], "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertEqual(meta["depth_convention"], ob.DEPTH_CONVENTION)
            self.assertEqual(meta["yaw_convention"], "habitat")


# --------------------------------------------------------------------------- #
# headless-Blender integration
# --------------------------------------------------------------------------- #
class BlenderObservationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if trimesh is None:
            raise unittest.SkipTest("trimesh is not installed")
        try:
            ob.find_blender()
        except FileNotFoundError as error:
            raise unittest.SkipTest(str(error))
        cls.tmp = tempfile.mkdtemp(prefix="roomgen_nav_obs_test_")
        cls.scene_dir = os.path.join(cls.tmp, "scene")
        cls.glb = build_known_room(os.path.join(cls.scene_dir, "build", "scene.glb"))
        cls.frame_dir = os.path.join(cls.tmp, "frames")
        cls.backend = ob.ObservationBackend(
            cls.scene_dir, samples=8, frame_dir=cls.frame_dir, timeout=120.0,
            yaw_convention="habitat")

    @classmethod
    def tearDownClass(cls):
        backend = getattr(cls, "backend", None)
        if backend is not None:
            backend.close()
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def test_default_frame_contract(self):
        observation = self.backend.get_observation(
            (0.0, 0.0, 0.0), 0.0, 0.0, (128, 128))
        self.assertEqual(observation.rgb.shape, (128, 128, 3))
        self.assertEqual(observation.rgb.dtype, np.uint8)
        self.assertEqual(observation.depth.shape, (128, 128))
        self.assertEqual(observation.depth.dtype, np.float32)
        self.assertEqual(observation.distance.shape, (128, 128))
        self.assertEqual(observation.distance.dtype, np.float32)
        self.assertEqual(observation.resolution, (128, 128))
        self.assertTrue(observation.valid.all(), "the camera is inside a closed room")
        self.assertGreater(float(observation.rgb.std()), 5.0, "RGB must not be a flat frame")
        self.assertGreater(float(observation.rgb.mean()), 5.0, "RGB must not be black")
        self.assertAlmostEqual(float(observation.intrinsics[0, 0]), 64.0, places=12)
        np.testing.assert_allclose(observation.extrinsics[:3, 3], (0.0, 0.0, 0.0))

    def test_centre_depth_matches_known_wall(self):
        observation = self.backend.get_observation(
            (0.0, 0.0, 0.0), 0.0, 0.0, (128, 128))
        centre = observation.depth[64, 64]
        self.assertAlmostEqual(float(centre), ROOM_HALF_Z, delta=0.01)
        # The centre pixel is half a pixel off the optical axis; the Euclidean
        # distance must exceed the z-depth by exactly that tiny factor.
        fx = observation.intrinsics[0, 0]
        ndc = math.hypot(0.5 / fx, 0.5 / fx)
        expected_distance = float(centre) * math.sqrt(1.0 + ndc * ndc)
        self.assertAlmostEqual(float(observation.distance[64, 64]),
                               expected_distance, delta=0.001)

    def test_off_axis_projection_uses_fx(self):
        observation = self.backend.get_observation(
            (0.0, 0.0, 0.0), 0.0, 0.0, (128, 128))
        fx = observation.intrinsics[0, 0]
        expected_column = 64.0 + fx * MARKER_X / (MARKER_Z - 0.01)
        self.assertAlmostEqual(expected_column, 96.65, delta=0.1)
        # The marker is green; the wall behind it is red.  A mirrored projection
        # would put the green marker around column 31 instead.
        patch = observation.rgb[60:69, 94:100].reshape(-1, 3).mean(axis=0)
        mirror = observation.rgb[60:69, 28:34].reshape(-1, 3).mean(axis=0)
        self.assertGreater(patch[1], patch[0] + 20,
                           f"marker should be green at column ~96, got {patch}")
        self.assertGreater(patch[1], patch[2] + 20,
                           f"marker should be green at column ~96, got {patch}")
        self.assertGreater(mirror[0], mirror[1] + 20,
                           f"left half should be the red wall, got {mirror}")
        self.assertAlmostEqual(float(observation.depth[64, 96]), MARKER_Z - 0.01,
                               delta=0.01)

    def test_yaw_turns_towards_the_expected_wall(self):
        forward = self.backend.get_observation((0.0, 0.0, 0.0), 0.0, 0.0, (128, 128))
        left = self.backend.get_observation((0.0, 0.0, 0.0), 90.0, 0.0, (128, 128))
        patch = (slice(60, 68), slice(60, 68))
        red = forward.rgb[patch].reshape(-1, 3).mean(axis=0)
        green = left.rgb[patch].reshape(-1, 3).mean(axis=0)
        self.assertGreater(red[0], red[1] + 20, f"yaw 0 must see the red -Z wall, got {red}")
        self.assertGreater(red[0], red[2] + 20, f"yaw 0 must see the red -Z wall, got {red}")
        self.assertGreater(green[1], green[0] + 20, f"yaw 90 must see the green -X wall, got {green}")
        self.assertGreater(green[1], green[2] + 20, f"yaw 90 must see the green -X wall, got {green}")
        self.assertAlmostEqual(float(forward.depth[64, 64]), ROOM_HALF_Z, delta=0.01)
        self.assertAlmostEqual(float(left.depth[64, 64]), ROOM_HALF_X, delta=0.01)

    def test_pitch_hits_floor_and_ceiling(self):
        # 63 is odd, so pixel (31, 31) samples exactly the optical axis.
        down = self.backend.get_observation((0.0, 0.0, 0.0), 0.0, -45.0, (63, 63))
        up = self.backend.get_observation((0.0, 0.0, 0.0), 0.0, 45.0, (63, 63))
        expected = ROOM_HALF_Y / math.cos(math.radians(45.0))
        self.assertAlmostEqual(float(down.depth[31, 31]), expected, delta=0.01)
        self.assertAlmostEqual(float(up.depth[31, 31]), expected, delta=0.01)
        self.assertAlmostEqual(float(down.distance[31, 31]), float(down.depth[31, 31]),
                               delta=1e-4)
        self.assertAlmostEqual(float(up.distance[31, 31]), float(up.depth[31, 31]),
                               delta=1e-4)

    def test_depth_and_distance_are_consistent(self):
        observation = self.backend.get_observation(
            (0.0, 0.0, 0.0), 20.0, -10.0, (64, 64))
        height, width = observation.depth.shape
        fx = observation.intrinsics[0, 0]
        columns = np.arange(width)[None, :] + 0.5 - width / 2.0
        rows = np.arange(height)[:, None] + 0.5 - height / 2.0
        factor = np.sqrt(1.0 + (columns / fx) ** 2 + (rows / fx) ** 2)
        expected = observation.depth * factor
        valid = observation.valid
        self.assertTrue(valid.all())
        np.testing.assert_allclose(observation.distance[valid], expected[valid],
                                   rtol=2e-5, atol=1e-4)

    def test_no_hit_is_zero(self):
        observation = self.backend.get_observation(
            (0.0, 10.0, 0.0), 0.0, 80.0, (32, 32))
        self.assertFalse(observation.valid.any())
        self.assertEqual(float(observation.depth.max()), 0.0)
        self.assertEqual(float(observation.distance.max()), 0.0)

    def test_same_pose_is_reproducible(self):
        first = self.backend.get_observation((0.0, 0.0, 0.0), 10.0, -5.0, (64, 64))
        second = self.backend.get_observation((0.0, 0.0, 0.0), 10.0, -5.0, (64, 64))
        np.testing.assert_array_equal(first.rgb, second.rgb)
        np.testing.assert_array_equal(first.depth, second.depth)
        np.testing.assert_array_equal(first.distance, second.distance)

    def test_reset_and_final_capture_files(self):
        reset = self.backend.reset((0.0, 0.0, 0.0), 0.0, 0.0, (64, 64))
        final = self.backend.final((0.0, 0.0, 0.0), 90.0, 0.0, (64, 64))
        self.assertIsNotNone(reset.saved_paths)
        self.assertIsNotNone(final.saved_paths)
        for observation, tag in ((reset, "reset"), (final, "final")):
            paths = observation.saved_paths
            self.assertEqual(os.path.basename(paths["rgb"]), f"{tag}_rgb.png")
            self.assertTrue(os.path.isfile(paths["rgb"]))
            self.assertTrue(os.path.isfile(paths["depth"]))
            self.assertTrue(os.path.isfile(paths["distance"]))
            self.assertTrue(os.path.isfile(paths["meta"]))
            np.testing.assert_allclose(np.load(paths["depth"]), observation.depth)
            with open(paths["meta"], "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertEqual(meta["intrinsics"], observation.intrinsics.tolist())
        # Different yaw really changes the frame.
        self.assertAlmostEqual(float(reset.depth[32, 32]), ROOM_HALF_Z, delta=0.01)
        self.assertAlmostEqual(float(final.depth[32, 32]), ROOM_HALF_X, delta=0.01)

    def test_manifest_yaw_convention(self):
        with ob.ObservationBackend(self.scene_dir, samples=8,
                                   yaw_convention="manifest", timeout=120.0) as backend:
            observation = backend.get_observation((0.0, 0.0, 0.0), 0.0, 0.0, (64, 64))
        # manifest yaw 0 -> +X, the grey wall at x = +3
        self.assertAlmostEqual(float(observation.depth[32, 32]), ROOM_HALF_X, delta=0.01)
        self.assertEqual(observation.yaw_convention, "manifest")

    def test_module_level_get_observation(self):
        try:
            observation = ob.get_observation(
                self.scene_dir, (0.0, 0.0, 0.0), 0.0,
                resolution=(32, 32), samples=4)
        finally:
            ob.close_backends()
        self.assertEqual(observation.rgb.shape, (32, 32, 3))
        self.assertEqual(observation.yaw_convention, "manifest")
        # Default convention is the project's compass heading: yaw 0 -> +X.
        self.assertAlmostEqual(float(observation.depth[16, 16]), ROOM_HALF_X, delta=0.01)

    def test_default_is_project_manifest(self):
        self.assertEqual(ob.DEFAULT_YAW_CONVENTION, "manifest")
        with ob.ObservationBackend(self.scene_dir, samples=4, timeout=120.0) as backend:
            observation = backend.get_observation((0.0, 0.0, 0.0), 0.0, 0.0, (63, 63))
        self.assertEqual(observation.yaw_convention, "manifest")
        # 63 is odd: pixel (31, 31) is the optical axis, looking at the grey +X wall.
        self.assertAlmostEqual(float(observation.depth[31, 31]), ROOM_HALF_X, delta=0.01)

    def test_env_provider_adapter(self):
        class FakeEnv:
            position = (0.0, -ROOM_HALF_Y, 0.0)  # body on the walking surface
            yaw_deg = 90.0  # manifest heading -Z

        with ob.ObservationBackend(self.scene_dir, samples=4, timeout=120.0) as backend:
            provider = ob.env_observation_provider(
                backend, pitch=-45.0, resolution=(63, 63), eye_height=ROOM_HALF_Y)
            result = provider(FakeEnv())
        self.assertIsInstance(result, dict)
        self.assertEqual(result["provider"], "blender_cycles")
        self.assertEqual(result["rgb"].shape, (63, 63, 3))
        self.assertEqual(result["depth"].shape, (63, 63))
        self.assertEqual(result["intrinsics"].shape, (3, 3))
        # eye_height lifts the camera from y=-1.5 to y=0; looking down 45 deg
        # then reaches the floor at 1.5 / cos(45 deg), not immediately.
        self.assertAlmostEqual(float(result["depth"][31, 31]),
                               ROOM_HALF_Y / math.cos(math.radians(45.0)), delta=0.01)
        self.assertAlmostEqual(float(result["position"][1]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
