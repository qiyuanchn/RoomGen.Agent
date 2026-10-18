"""Unit tests for the pure navigation inventory / mesh helpers (host Python)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import spec as spec_mod  # noqa: E402
from roomgen_nav import assets  # noqa: E402

DEMO_SPEC = os.path.join(TESTS_DIR, "fixtures", "two_floor_demo.json")


def _box(x0, y0, z0, x1, y1, z1):
    """Return (vertices, triangles) for an axis-aligned box in Blender Z-up."""
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return corners, faces


def _mesh_from_boxes(boxes):
    vertices = []
    triangles = []
    base = 0
    for box in boxes:
        points, faces = _box(*box)
        vertices.append(points)
        triangles.append(faces + base)
        base += len(points)
    return assets.Mesh(np.concatenate(vertices), np.concatenate(triangles))


class MeshIOTest(unittest.TestCase):
    def test_ply_roundtrip(self):
        mesh = _mesh_from_boxes([(0, 0, 0, 2, 1, 1)])
        with self.subTest("write/read"):
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "mesh.ply")
                stats = assets.write_ply(path, mesh)
                self.assertEqual(stats["triangles"], 12)
                again = assets.read_ply(path)
                self.assertEqual(again.triangles.shape, (12, 3))
                np.testing.assert_allclose(again.vertices, mesh.vertices, atol=1e-6)

    def test_floor_extraction_keeps_the_right_storey(self):
        spec = spec_mod.load_spec(DEMO_SPEC)
        floors = {floor.id: floor for floor in spec.floors}
        # Two slabs plus a piece of lower-floor furniture; the upper slab must
        # not leak into the lower band.
        mesh = _mesh_from_boxes([
            (-0.5, -0.5, -0.15, 8.5, 6.5, 0.0),        # f0 slab
            (0.0, 0.0, 1.0, 1.0, 1.0, 1.5),            # f0 furniture
            (-0.5, -0.5, 2.70, 8.5, 6.5, 2.85),        # f1 slab
            (0.0, 0.0, 4.0, 1.0, 1.0, 4.5),            # f1 furniture
        ])
        lower = assets.extract_floor_mesh(mesh, spec, floors["f0"])
        upper = assets.extract_floor_mesh(mesh, spec, floors["f1"])
        self.assertEqual(len(lower), 24)  # two boxes
        self.assertEqual(len(upper), 24)
        lo, hi = assets.floor_band(spec, floors["f0"])
        self.assertLess(lo, 0.0)
        self.assertLess(hi, 2.85)
        self.assertTrue(np.all(lower.vertices[..., 2] <= 2.85))
        self.assertTrue(np.all(upper.vertices[..., 2] >= 2.70))

    def test_write_obj_uses_habitat_y_up(self):
        import tempfile
        mesh = _mesh_from_boxes([(0, 0, 0, 1, 1, 1)])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "floor.obj")
            stats = assets.write_floor_obj(mesh, path)
            with open(path, "r", encoding="ascii") as handle:
                first = handle.readline().strip().split()
            self.assertEqual(first[0], "v")
            # Blender (x, y, z) -> (x, z, -y): the first vertex (0,0,0)
            self.assertEqual([float(v) for v in first[1:]], [0.0, 0.0, 0.0])
            self.assertEqual(stats["triangles"], 12)

    def test_glb_reader_matches_a_known_export(self):
        glb = os.path.join(REPO_ROOT, "outputs", "viewer_walk_duplex", "build", "scene.glb")
        if not os.path.isfile(glb):
            self.skipTest("viewer_walk_duplex GLB not present")
        mesh = assets.read_glb(glb)
        self.assertGreater(len(mesh), 1000)
        lo, hi = mesh.bounds
        # Blender Z-up: floors at 0 and 2.9, slab and furniture above it.
        self.assertLess(hi[2], 20.0)
        self.assertGreater(hi[2], 2.0)

    def test_scene_bounds_and_inventory(self):
        spec = spec_mod.load_spec(DEMO_SPEC)
        lo, hi = assets.scene_bounds(spec)
        self.assertLess(lo[0], 0.0)
        self.assertGreater(hi[0], 5.0)
        entries = assets.default_scenes()
        self.assertEqual(len(entries), 20)
        self.assertEqual(len({entry.scene_id for entry in entries}), 20)
        self.assertTrue(all(os.path.isabs(entry.spec_path) for entry in entries))


if __name__ == "__main__":
    unittest.main()
