#!/usr/bin/env python3
"""Node-executed geometry proof for the browser Walk mode.

`tools/scene3d_viewer/walk.mjs` is dependency-free on purpose: the viewer feeds
it the loaded GLB triangles, and this test builds small triangle scenes (slabs,
a stair, a slab opening, a wall, furniture) and drives the same `stepWalk`.
The static checks below pin the wiring: viewer.js must import the module and
must not snap the eye to a fixed constant.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_TEST = os.path.join(REPO, "tests", "walk_geometry.test.mjs")
VIEWER = os.path.join(REPO, "tools", "scene3d_viewer", "viewer.js")


class WalkGeometryTest(unittest.TestCase):
    def test_node_geometry_suite(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        result = subprocess.run(
            [node, NODE_TEST],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("walk geometry:", result.stdout)

    def test_viewer_uses_the_glb_support_index(self):
        with open(VIEWER, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("from './walk.mjs'", source)
        self.assertIn("stepWalk(state.walkIndex", source)
        self.assertIn("supportBelow(state.walkIndex", source)
        self.assertIn("findStandingSpot(state.walkIndex", source)
        self.assertNotIn("camera.position.y = state.eyeHeight", source)

    def test_floor_selector_and_section_wiring(self):
        with open(VIEWER, "r", encoding="utf-8") as handle:
            source = handle.read()
        index_path = os.path.join(REPO, "tools", "scene3d_viewer", "index.html")
        with open(index_path, "r", encoding="utf-8") as handle:
            page = handle.read()
        self.assertIn('id="floor-select"', page)
        self.assertIn('id="section-toggle"', page)
        self.assertIn("function applyClipping", source)
        self.assertIn("new THREE.Plane", source)
        self.assertIn("setActiveFloor", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
