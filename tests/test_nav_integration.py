"""End-to-end integration: bake a duplex, link the floors, replay an episode.

The host test interpreter has no Habitat-Sim, so the real work runs in the
``third_party/nav_env`` interpreter as a subprocess.  The test covers the exact
acceptance claim: a two-storey scene produces a cross-floor path, the stored
actions replay successfully, and the wheeled profile is rejected by the stair
riser instead of silently sharing the humanoid NavMesh.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from roomgen_nav import assets  # noqa: E402
VIEWER_BUILD = os.path.join(REPO_ROOT, "outputs", "viewer_walk_duplex", "build")
NAV_PYTHON = os.environ.get(
    "ROOMGEN_NAV_PYTHON", os.path.join(REPO_ROOT, "third_party", "nav_env", "bin", "python"))
NAV_LIB = os.path.join(REPO_ROOT, "third_party", "nav_sysroot", "usr", "lib",
                       "x86_64-linux-gnu")
NAV_ENV_LIB = os.path.join(REPO_ROOT, "third_party", "nav_env", "lib")

SCRIPT = r"""
import json, os, random, sys, tempfile
sys.path.insert(0, sys.argv[1])
from roomgen_nav import assets, profiles, episodes
from roomgen_nav.navmesh import build_composite_navmesh

base = sys.argv[2]
entry = assets.SceneEntry("viewer_walk_duplex",
                          os.path.join(base, "build_manifest.json"), base)
spec = assets.load_scene(entry)
result = {}
with tempfile.TemporaryDirectory() as cache:
    nav = build_composite_navmesh(entry, profiles.HUMANOID, cache, spec=spec, rebuild=True)
    catalog = episodes.generate_catalog("viewer_walk_duplex", spec, profiles.HUMANOID,
                                        nav, count=3, base_seed=20261018)
    verification = episodes.verify_catalog("viewer_walk_duplex", spec, profiles.HUMANOID,
                                           nav, catalog)
    cross = [ep for ep in catalog.episodes if ep.cross_floor]
    result["humanoid"] = {
        "stats": nav.stats(),
        "links": [link.to_dict() for link in nav.links],
        "generated": len(catalog),
        "report": catalog.report,
        "verification": verification,
        "cross_floor": len(cross),
        "cross_actions": len(cross[0].actions) if cross else 0,
        "cross_success": bool(cross and all(ep.reference_success for ep in cross)),
        "cross_links": cross[0].links_used if cross else [],
        "all_success": all(ep.reference_success for ep in catalog.episodes),
    }
    glb_entry = assets.SceneEntry(
        "viewer_walk_duplex_glb",
        os.path.join(base, "build_manifest.json"), base,
        mesh_path=os.path.join(base, "scene.glb"))
    nav_glb = build_composite_navmesh(glb_entry, profiles.HUMANOID,
                                     os.path.join(cache, "glb"), spec=spec, rebuild=True)
    glb_rng = random.Random(1)
    glb_start = nav_glb.sample_room_point("living", glb_rng)
    glb_goal = nav_glb.sample_room_point("bedroom", glb_rng)
    glb_path = nav_glb.find_path("F1", glb_start, "F2", glb_goal) \
        if glb_start and glb_goal else None
    result["glb"] = {
        "source": assets.resolve_mesh(glb_entry),
        "stats": nav_glb.stats(),
        "start": glb_start,
        "goal": glb_goal,
        "has_path": glb_path is not None,
        "links": glb_path.links_used if glb_path else [],
    }
    nav_w = build_composite_navmesh(entry, profiles.WHEELED, os.path.join(cache, "wheeled"),
                                    spec=spec, rebuild=True)
    catalog_w = episodes.generate_catalog("viewer_walk_duplex", spec, profiles.WHEELED,
                                          nav_w, count=3, base_seed=20261018)
    result["wheeled"] = {
        "stats": nav_w.stats(),
        "links": [link.to_dict() for link in nav_w.links],
        "generated": len(catalog_w),
        "cross_floor": sum(1 for ep in catalog_w.episodes if ep.cross_floor),
        "all_success": all(ep.reference_success for ep in catalog_w.episodes),
    }
print("RESULT:" + json.dumps(result))
"""


@unittest.skipUnless(os.path.isfile(os.path.join(VIEWER_BUILD, "build_manifest.json")),
                     "viewer_walk_duplex build is not present")
@unittest.skipUnless(os.path.isfile(NAV_PYTHON), "third_party/nav_env is not present")
class DuplexIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = NAV_LIB + ":" + NAV_ENV_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
        completed = subprocess.run(
            [NAV_PYTHON, "-c", SCRIPT, REPO_ROOT, VIEWER_BUILD],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=240)
        cls.stderr = completed.stderr
        cls.stdout = completed.stdout
        cls.payload = None
        for line in completed.stdout.splitlines():
            if line.startswith("RESULT:"):
                cls.payload = json.loads(line[len("RESULT:"):])
        if completed.returncode != 0 or cls.payload is None:
            raise AssertionError(
                "nav integration subprocess failed\nSTDOUT:\n"
                + completed.stdout[-4000:] + "\nSTDERR:\n" + completed.stderr[-4000:])

    def test_humanoid_crosses_the_stair(self):
        humanoid = self.payload["humanoid"]
        self.assertEqual(humanoid["stats"]["floors"], 2)
        self.assertGreaterEqual(humanoid["stats"]["valid_links"], 1)
        self.assertGreaterEqual(humanoid["cross_floor"], 1)
        self.assertTrue(humanoid["cross_success"])
        self.assertTrue(humanoid["all_success"])
        self.assertTrue(any(link.startswith("stair_") for link in humanoid["cross_links"]))
        self.assertGreater(humanoid["cross_actions"], 20)
        self.assertTrue(humanoid["verification"]["ok"])

    def test_wheeled_cannot_use_the_stair(self):
        wheeled = self.payload["wheeled"]
        self.assertEqual(wheeled["stats"]["valid_links"], 0)
        self.assertTrue(any("riser" in link["reason"] for link in wheeled["links"]))
        self.assertEqual(wheeled["cross_floor"], 0)
        self.assertTrue(wheeled["all_success"])
        self.assertGreaterEqual(wheeled["generated"], 1)

    def test_glb_source_builds_the_two_floors(self):
        glb = self.payload["glb"]
        self.assertTrue(glb["source"].endswith("scene.glb"))
        self.assertEqual(glb["stats"]["floors"], 2)
        self.assertEqual(glb["stats"]["valid_links"], 1)
        self.assertTrue(glb["has_path"])
        self.assertTrue(any(link.startswith("stair_") for link in glb["links"]))


def _box(x0, y0, z0, x1, y1, z1):
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


def _door_scene(directory: str, gap: float = 0.30) -> str:
    """Write a two-room shell whose only connection is a ``gap`` metre door."""
    y0, y1 = 1.5 - gap / 2.0, 1.5 + gap / 2.0
    boxes = [
        (0, 0, -0.1, 6, 3, 0),                       # floor
        (0, 0, 0, 6, 0.1, 2.7),                      # north wall
        (0, 2.9, 0, 6, 3, 2.7),                      # south wall
        (0, 0, 0, 0.1, 3, 2.7),                      # west wall
        (5.9, 0, 0, 6, 3, 2.7),                      # east wall
        (2.95, 0, 0, 3.05, y0, 2.7),                 # internal wall, below door
        (2.95, y1, 0, 3.05, 3, 2.7),                 # internal wall, above door
    ]
    vertices, triangles, base = [], [], 0
    for values in boxes:
        points, faces = _box(*values)
        vertices.append(points)
        triangles.append(faces + base)
        base += len(points)
    mesh = assets.Mesh(np.concatenate(vertices), np.concatenate(triangles))
    assets.write_ply(os.path.join(directory, "furnished.ply"), mesh)
    spec = {
        "unit": "meter", "wall_height": 2.7, "wall_thickness": 0.1,
        "rooms": [
            {"id": "a", "type": "room",
             "polygon": [[0, 0], [3, 0], [3, 3], [0, 3]]},
            {"id": "b", "type": "room",
             "polygon": [[3, 0], [6, 0], [6, 3], [3, 3]]},
        ],
        "cameras": [{"name": "c", "room": "a", "pos": [1.5, 1.5, 1.5]}],
        "openings": [{"kind": "door", "between": ["a", "b"],
                      "center": [3.0, 1.5], "width": gap, "height": 2.05}],
    }
    path = os.path.join(directory, "spec.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle)
    return directory


DOOR_SCRIPT = r"""
import json, os, sys, tempfile, random
sys.path.insert(0, sys.argv[1])
from roomgen_nav import assets, profiles
from roomgen_nav.navmesh import build_composite_navmesh

directory = sys.argv[2]
entry = assets.SceneEntry("narrow_door", os.path.join(directory, "spec.json"),
                          directory)
spec = assets.load_scene(entry)
result = {}
profiles_to_test = {
    "humanoid": profiles.HUMANOID,
    "tiny": profiles.AgentProfile("tiny", radius=0.10, height=1.0,
                                  max_step=0.25, max_slope_deg=45.0),
}
for name, profile in profiles_to_test.items():
    with tempfile.TemporaryDirectory() as cache:
        nav = build_composite_navmesh(entry, profile, cache, spec=spec, rebuild=True)
        rng = random.Random(1)
        start = nav.sample_room_point("a", rng)
        goal = nav.sample_room_point("b", rng)
        path = nav.find_path("F0", start, "F0", goal) if start and goal else None
        result[name] = {
            "stats": nav.stats(),
            "start": start,
            "goal": goal,
            "has_path": path is not None,
            "distance": path.distance if path else None,
        }
print("RESULT:" + json.dumps(result))
"""


@unittest.skipUnless(os.path.isfile(NAV_PYTHON), "third_party/nav_env is not present")
class NarrowDoorIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp(prefix="roomgen_nav_door_")
        _door_scene(cls.directory, gap=0.30)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = NAV_LIB + ":" + NAV_ENV_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
        completed = subprocess.run(
            [NAV_PYTHON, "-c", DOOR_SCRIPT, REPO_ROOT, cls.directory],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180)
        cls.payload = None
        for line in completed.stdout.splitlines():
            if line.startswith("RESULT:"):
                cls.payload = json.loads(line[len("RESULT:"):])
        if completed.returncode != 0 or cls.payload is None:
            raise AssertionError(
                "narrow-door nav subprocess failed\nSTDOUT:\n"
                + completed.stdout[-4000:] + "\nSTDERR:\n" + completed.stderr[-4000:])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_humanoid_is_blocked_by_a_narrow_door(self):
        humanoid = self.payload["humanoid"]
        self.assertTrue(humanoid["start"] and humanoid["goal"])
        self.assertFalse(humanoid["has_path"])
        self.assertEqual(humanoid["stats"]["islands"]["F0"], 2)

    def test_smaller_agent_passes_the_same_door(self):
        tiny = self.payload["tiny"]
        self.assertTrue(tiny["has_path"])
        self.assertGreater(tiny["distance"], 1.0)
        self.assertEqual(tiny["stats"]["islands"]["F0"], 1)


if __name__ == "__main__":
    unittest.main()
