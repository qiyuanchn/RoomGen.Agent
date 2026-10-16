"""Numerical unit tests for the 3D-FRONT replay importer (host python, no bpy).

Run::

    python tests/test_front3d_math.py            # or: python tests/test_front3d_math.py -v

Everything here is pinned against hand-computed numbers: the quaternion
convention, the order in which scale/rotation/translation are applied, the
Y-up -> Z-up frame conversion, the base-centre model origin, the room outline
recovered from the fixture's floor mesh, and the bridge back to a scene spec
that ``spec.py`` accepts.  The fixture is ``tests/fixtures/front3d_mini`` (see
``make_fixture.py`` for its hand-authored geometry and the expected poses).
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import front3d, spec as spec_mod  # noqa: E402

FIXTURE = os.path.join(TESTS_DIR, "fixtures", "front3d_mini")
SCENE_JSON = os.path.join(FIXTURE, "scene.json")
MODEL_ROOT = os.path.join(FIXTURE, "3D-FUTURE-model")
TEXTURE_ROOT = os.path.join(FIXTURE, "3D-FRONT-textures")

SQRT_HALF = math.sqrt(0.5)
YAW90 = [0.0, SQRT_HALF, 0.0, SQRT_HALF]           # 90 deg about the dataset's +Y
BED_JID = "0f1b6f2e-1c1e-4a5a-9b1f-0000000000b2"
MISSING_JID = "0f1b6f2e-1c1e-4a5a-9b1f-ffffffffffff"


def load_scene():
    return front3d.parse_scene(SCENE_JSON, model_root=MODEL_ROOT)


# --------------------------------------------------------------------------- #
# 1. quaternion convention: (x, y, z, w), rotation about the dataset's up axis
# --------------------------------------------------------------------------- #
class QuaternionTests(unittest.TestCase):
    def test_identity(self):
        identity = front3d.quaternion_xyzw_to_matrix([0.0, 0.0, 0.0, 1.0])
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(identity[i][j], 1.0 if i == j else 0.0, places=12)

    def test_yaw_about_up_axis_matches_hand_computed(self):
        """q = (0, sin45, 0, cos45) is +90 deg about the dataset's +Y.

        Hand-computed: (x, y, z) -> (z, y, -x).
        """
        matrix = front3d.quaternion_xyzw_to_matrix(YAW90)
        expected = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(matrix[i][j], expected[i][j], places=12)
        for vector, rotated in (((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
                                ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
                                ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0))):
            got = [sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3)]
            for i in range(3):
                self.assertAlmostEqual(got[i], rotated[i], places=12)

    def test_is_a_proper_rotation(self):
        matrix = front3d.quaternion_xyzw_to_matrix(YAW90)
        determinant = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
        self.assertAlmostEqual(determinant, 1.0, places=12)

    def test_wxyz_order_would_be_a_different_rotation(self):
        """The BlenderProc trap: ``(w, x, y, z)`` reading of the same list."""
        as_wxyz = front3d.quaternion_xyzw_to_matrix([YAW90[3], YAW90[0], YAW90[1], YAW90[2]])
        as_xyzw = front3d.quaternion_xyzw_to_matrix(YAW90)
        difference = max(abs(as_wxyz[i][j] - as_xyzw[i][j]) for i in range(3) for j in range(3))
        self.assertGreater(difference, 0.5)

    def test_yaw_extraction(self):
        yaw, off_axis = front3d.yaw_deg_from_quaternion(YAW90)
        self.assertAlmostEqual(yaw, 90.0, places=9)
        self.assertLess(off_axis, 1e-12)
        yaw, off_axis = front3d.yaw_deg_from_quaternion([0.0, 0.0, 0.0, 1.0])
        self.assertAlmostEqual(yaw, 0.0, places=12)


# --------------------------------------------------------------------------- #
# 2. transform order: scale (per axis), then rotate, then translate
# --------------------------------------------------------------------------- #
class TransformOrderTests(unittest.TestCase):
    POS = [5.4, 0.0, 1.7]
    SCALE = [1.0, 0.8, 1.1]
    POINT = [0.8, 1.1, 1.0]

    def hand_computed(self):
        """scale -> yaw 90 about +Y -> translate, written out by hand."""
        scaled = [self.POINT[i] * self.SCALE[i] for i in range(3)]
        rotated = [scaled[2], scaled[1], -scaled[0]]
        return [rotated[i] + self.POS[i] for i in range(3)]

    def test_dataset_order(self):
        matrix = front3d.scene_instance_matrix(self.POS, YAW90, self.SCALE, frame=False)
        got = front3d.mat4_apply(matrix, self.POINT)
        expected = self.hand_computed()
        for i in range(3):
            self.assertAlmostEqual(got[i], expected[i], places=12)

    def test_wrong_orders_are_different(self):
        """Guard rails: the two plausible mistakes give different results."""
        expected = self.hand_computed()
        # (a) translate before rotating
        shifted = [self.POINT[i] + self.POS[i] for i in range(3)]
        translate_first = [shifted[2], shifted[1], -shifted[0]]
        # (b) scale after rotating
        rotated = [self.POINT[2], self.POINT[1], -self.POINT[0]]
        scale_last = [rotated[i] * self.SCALE[i] + self.POS[i] for i in range(3)]
        self.assertGreater(max(abs(translate_first[i] - expected[i]) for i in range(3)), 0.1)
        self.assertGreater(max(abs(scale_last[i] - expected[i]) for i in range(3)), 0.05)

    def test_uniform_scale_equals_vector_scale(self):
        matrix = front3d.scene_instance_matrix(self.POS, YAW90, [1.3, 1.3, 1.3], frame=False)
        got = front3d.mat4_apply(matrix, self.POINT)
        scaled = [value * 1.3 for value in self.POINT]
        expected = [scaled[2] + self.POS[0], scaled[1] + self.POS[1], -scaled[0] + self.POS[2]]
        for i in range(3):
            self.assertAlmostEqual(got[i], expected[i], places=12)

    def test_scale_xyz_is_rejected_when_malformed(self):
        with self.assertRaises(ValueError):
            front3d.scene_instance_matrix(self.POS, YAW90, [1.0, 1.0])


# --------------------------------------------------------------------------- #
# 3. Y-up (dataset) -> Z-up (Blender / repo) conversion
# --------------------------------------------------------------------------- #
class FrameConversionTests(unittest.TestCase):
    def test_point_mapping(self):
        self.assertEqual(front3d.front_to_blender_point([1.0, 2.0, 3.0]), [1.0, -3.0, 2.0])
        self.assertEqual(front3d.blender_to_front_point([1.0, -3.0, 2.0]), [1.0, 2.0, 3.0])

    def test_basis_vectors(self):
        matrix = front3d.FRONT_TO_BLENDER
        for vector, expected in (((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),   # +X stays right
                                 ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),   # up: +Y -> +Z
                                 ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0))):  # model front -> -Y
            got = front3d.mat4_apply(matrix, vector)
            for i in range(3):
                self.assertAlmostEqual(got[i], expected[i], places=12)

    def test_is_a_proper_rotation_not_a_mirror(self):
        """BlenderProc swaps y/z (a mirror); we rotate, so winding survives."""
        matrix = front3d.FRONT_TO_BLENDER
        determinant = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
        self.assertAlmostEqual(determinant, 1.0, places=12)

    def test_round_trip_through_the_fixture_spec(self):
        """spec item -> world matrix must reproduce the dataset's own matrix."""
        scene = load_scene()
        spec = front3d.to_scene_spec(scene, model_root=MODEL_ROOT, texture_root=TEXTURE_ROOT)
        instances = {instance.uid: instance for instance in scene.instances}
        checked = 0
        for item in spec["furniture"]:
            instance = instances[item["name"]]
            rebuilt = front3d.item_matrix(item["pos"], item["z"], item["rot_z_deg"], item["scale_xyz"])
            original = instance.matrix(frame=True)
            worst = max(
                abs(rebuilt[i][j] - original[i][j]) for i in range(4) for j in range(4)
            )
            self.assertLess(worst, 1e-9, f"{item['name']} pose round-trip drifted by {worst}")
            checked += 1
        self.assertEqual(checked, 5)


# --------------------------------------------------------------------------- #
# 4. the fixture itself: shell geometry, rooms, models
# --------------------------------------------------------------------------- #
class FixtureParseTests(unittest.TestCase):
    def setUp(self):
        self.scene = load_scene()

    def test_counts(self):
        self.assertEqual(self.scene.uid, "front3d-mini-0001")
        self.assertEqual(self.scene.version, "2.0")
        self.assertEqual(len(self.scene.meshes), 20)
        self.assertEqual(len(self.scene.materials), 5)
        self.assertEqual(len(self.scene.furniture), 7)
        self.assertEqual(len(self.scene.rooms), 2)
        # every furniture child is recorded (so the report can count the ones the
        # dataset flags invalid), but only the valid ones are placement records
        self.assertEqual(len(self.scene.instances), 7)
        self.assertEqual(len(self.scene.instance_records()), 6)
        self.assertEqual(self.scene.warnings, [])

    def test_wall_height_from_ceiling_meshes(self):
        self.assertAlmostEqual(front3d.shell_wall_height(self.scene), 2.6, places=9)

    def test_floor_mesh_maps_to_z_zero_and_ceiling_to_2_6(self):
        floor = [mesh for mesh in self.scene.meshes.values() if mesh.type == "Floor"]
        ceiling = [mesh for mesh in self.scene.meshes.values() if mesh.type == "Ceiling"]
        floor_z = {round(front3d.front_to_blender_point(point)[2], 9)
                   for mesh in floor for point in mesh.points()}
        ceiling_z = {round(front3d.front_to_blender_point(point)[2], 9)
                     for mesh in ceiling for point in mesh.points()}
        self.assertEqual(floor_z, {0.0})
        self.assertEqual(ceiling_z, {2.6})
        # the front-frame z is the repo-frame -y (a known pose in the fixture)
        north_wall = [mesh for mesh in self.scene.meshes.values() if mesh.constructid == "300102"][0]
        repo_y = {round(front3d.front_to_blender_point(point)[1], 9) for point in north_wall.points()}
        self.assertEqual(repo_y, {-3.2})

    def test_room_outlines_match_the_declared_size(self):
        widths = {"LivingDiningRoom-mini1": (0.0, 3.5), "BedRoom-mini2": (3.5, 7.0)}
        for room in self.scene.rooms:
            self.assertEqual(room.polygon_source, "floor_outline")
            # counter-clockwise for spec.py
            self.assertGreater(front3d.polygon_signed_area(room.polygon), 0.0)
            self.assertAlmostEqual(room.area, 11.2, places=6)      # 3.5 x 3.2 m
            self.assertAlmostEqual(float(room.size), room.area, places=6)
            xs = sorted(point[0] for point in room.polygon)
            ys = sorted(point[1] for point in room.polygon)
            self.assertAlmostEqual(xs[0], widths[room.instanceid][0], places=6)
            self.assertAlmostEqual(xs[-1], widths[room.instanceid][1], places=6)
            self.assertAlmostEqual(ys[0], -3.2, places=6)
            self.assertAlmostEqual(ys[-1], 0.0, places=6)

    def test_room_functions_from_categories(self):
        self.assertEqual(self.scene.room_functions["LivingDiningRoom-mini1"], "living")
        self.assertEqual(self.scene.room_functions["BedRoom-mini2"], "bedroom")
        self.assertEqual(self.scene.category_functions["sofa"], "living")
        self.assertEqual(self.scene.category_functions["bed"], "bedroom")

    def test_categories_are_case_folded_and_counted(self):
        self.assertEqual(self.scene.categories["sofa"], 2)
        self.assertEqual(self.scene.categories["bed"], 1)
        self.assertEqual(self.scene.categories["table"], 2)

    def test_invalid_furniture_is_skipped(self):
        self.assertFalse(self.scene.furniture["150/model"].valid)
        self.assertNotIn("150/model", [instance.uid for instance in self.scene.instance_records()])

    def test_cameras_sit_where_hand_computed(self):
        spec = front3d.to_scene_spec(self.scene, model_root=MODEL_ROOT)
        poses = {camera["name"]: camera["pos"] for camera in spec["cameras"]}
        self.assertEqual(len(poses), 2)
        for i in range(3):
            self.assertAlmostEqual(float(poses["0000"][i]), [1.75, -1.6, 1.5][i], places=9)
            self.assertAlmostEqual(float(poses["0001"][i]), [5.25, -1.6, 1.5][i], places=9)
        self.assertEqual(spec["cameras"][0]["room"], "LivingDiningRoom-mini1")

    def test_snapshot_camera_source(self):
        spec = front3d.to_scene_spec(self.scene, model_root=MODEL_ROOT, camera_source="snapshots")
        poses = [camera["pos"] for camera in spec["cameras"]]
        self.assertIn([1.75, -2.4, 1.5], [[round(v, 6) for v in pose] for pose in poses])

    def test_model_index_is_optional(self):
        scene = front3d.parse_scene(SCENE_JSON)          # no model root -> title categories
        self.assertEqual(scene.furniture["131/model"].category, "bed")
        self.assertEqual(scene.categories["sofa"], 2)


# --------------------------------------------------------------------------- ## 5. the spec bridge
# --------------------------------------------------------------------------- #
class SpecBridgeTests(unittest.TestCase):
    def setUp(self):
        self.scene = load_scene()
        self.spec_dict = front3d.to_scene_spec(self.scene, model_root=MODEL_ROOT,
                                               texture_root=TEXTURE_ROOT)

    def test_spec_module_accepts_the_bridge(self):
        spec = spec_mod.parse_spec(self.spec_dict, source_path="<bridge>")
        self.assertEqual(len(spec.rooms), 2)
        self.assertEqual(len(spec.cameras), 2)
        self.assertEqual(len(spec.furniture), 5)
        self.assertEqual(spec.front3d["shell"], "shell_mesh")
        self.assertEqual(spec.front3d["scene_uid"], self.scene.uid)
        self.assertAlmostEqual(spec.wall_height, 2.6, places=9)

    def test_front3d_items_keep_the_dataset_pose(self):
        bed = [item for item in self.spec_dict["furniture"] if item["name"] == "131/model"][0]
        self.assertEqual(bed["asset"], f"front3d:{BED_JID}")
        self.assertAlmostEqual(bed["pos"][0], 5.4, places=9)
        self.assertAlmostEqual(bed["pos"][1], -1.7, places=9)
        self.assertAlmostEqual(bed["z"], 0.0, places=9)
        self.assertAlmostEqual(bed["rot_z_deg"], 90.0, places=9)
        self.assertEqual(bed["scale_xyz"], [1.0, 0.8, 1.1])       # non-uniform, kept
        self.assertAlmostEqual(bed["scale"], 1.0, places=12)
        parsed = spec_mod.parse_spec(self.spec_dict)
        bed_spec = [item for item in parsed.furniture if item.name == "131/model"][0]
        self.assertEqual(bed_spec.scale_xyz, [1.0, 0.8, 1.1])

    def test_furniture_outside_every_room_is_reported_not_emitted(self):
        skipped = self.spec_dict["front3d"]["skipped_furniture"]
        self.assertEqual([entry["uid"] for entry in skipped], ["160/model"])
        self.assertEqual(skipped[0]["reason"], "outside every room outline")
        self.assertNotIn("160/model", [item["name"] for item in self.spec_dict["furniture"]])
        self.assertEqual(self.spec_dict["front3d"]["source"]["instances_placed"], 5)
        self.assertEqual(self.spec_dict["front3d"]["source"]["instances_dropped"], 1)

    def test_no_openings_or_windows_are_invented(self):
        self.assertEqual(self.spec_dict["openings"], [])
        self.assertEqual(self.spec_dict["windows"], [])
        self.assertEqual(self.spec_dict["details"],
                         {"skirting": False, "casings": False, "bevels": False, "glass": False})

    def test_spec_rejects_a_bad_front3d_block(self):
        broken = json.loads(json.dumps(self.spec_dict))
        broken["front3d"]["shell"] = "polygon_extrude"
        with self.assertRaises(spec_mod.SpecError):
            spec_mod.parse_spec(broken)
        broken = json.loads(json.dumps(self.spec_dict))
        del broken["front3d"]["scene_json"]
        with self.assertRaises(spec_mod.SpecError):
            spec_mod.parse_spec(broken)

    def test_spec_rejects_bad_scale_xyz(self):
        broken = json.loads(json.dumps(self.spec_dict))
        broken["furniture"][0]["scale_xyz"] = [1.0, 0.0, 1.0]
        with self.assertRaises(spec_mod.SpecError):
            spec_mod.parse_spec(broken)


# --------------------------------------------------------------------------- #
# 6. no Blender needed on the host side
# --------------------------------------------------------------------------- #
class HostPurityTests(unittest.TestCase):
    def test_importing_front3d_does_not_import_bpy(self):
        self.assertNotIn("bpy", sys.modules)

    def test_transform_helpers_are_stdout_free_pure_python(self):
        matrix = front3d.identity4()
        self.assertEqual(front3d.mat4_apply(matrix, [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])
        inverse = front3d.mat4_inverse_rotation(front3d.item_matrix([1.0, 2.0], 0.0, 90.0, [1.0, 1.0, 1.0]))
        product = front3d.mat4_mul(front3d.item_matrix([1.0, 2.0], 0.0, 90.0, [1.0, 1.0, 1.0]), inverse)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(product[i][j], 1.0 if i == j else 0.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
