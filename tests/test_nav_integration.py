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
from roomgen_nav.env import PointNavEnv, ShortestPathPolicy
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
    path_policy = None
    if cross:
        env = PointNavEnv("viewer_walk_duplex", spec, profiles.HUMANOID, nav,
                          auto_success=False)
        first = ShortestPathPolicy(env).run(cross[0])
        second = ShortestPathPolicy(env).run(cross[0])
        path_policy = {
            "success": bool(first["result"]["success"]),
            "spl": float(first["result"]["spl"]),
            "collisions": int(first["result"]["collisions"]),
            "steps": int(first["result"]["steps"]),
            "failure_reason": first["result"]["failure_reason"],
            "actions": first["actions"],
            "trajectory_length": len(first["trajectory"]),
            "reproducible": first["actions"] == second["actions"]
            and [step["position"] for step in first["trajectory"]]
            == [step["position"] for step in second["trajectory"]],
            "last_action": first["actions"][-1] if first["actions"] else None,
        }
    result["humanoid"]["path_policy"] = path_policy
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

    def test_path_policy_crosses_the_stair_closed_loop(self):
        policy = self.payload["humanoid"]["path_policy"]
        self.assertIsNotNone(policy)
        self.assertTrue(policy["success"])
        self.assertGreater(policy["steps"], 20)
        self.assertTrue(policy["reproducible"])
        self.assertEqual(policy["last_action"], "stop")
        self.assertEqual(policy["trajectory_length"], len(policy["actions"]) + 1)
        self.assertTrue(set(policy["actions"]).issubset(
            {"left", "right", "forward", "stop"}))
        self.assertTrue(any(action == "forward" for action in policy["actions"]))

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


BENCH_SCENE_BUILD = os.path.join(
    REPO_ROOT, "outputs", "benchmark100", "sample", "scene_0000", "build")


@unittest.skipUnless(os.path.isfile(os.path.join(BENCH_SCENE_BUILD, "build_manifest.json")),
                     "bench_scene_0000 build is not present")
@unittest.skipUnless(os.path.isfile(NAV_PYTHON), "third_party/nav_env is not present")
class PointNavEvalCliTest(unittest.TestCase):
    """The evaluator CLI writes separated policy / oracle results and JSONL."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="roomgen_nav_eval_")
        cls.out = os.path.join(cls.tmp, "out")
        cls.cache = os.path.join(cls.tmp, "cache")
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = (NAV_LIB + ":" + NAV_ENV_LIB + ":"
                                  + env.get("LD_LIBRARY_PATH", ""))
        cls.completed = subprocess.run(
            [NAV_PYTHON, "-m", "roomgen_nav.eval_pointnav",
             "--scene-ids", "bench_scene_0000", "--episodes", "2",
             "--policy", "path", "--repro-check", "--verify-sample", "0",
             "--out", cls.out, "--cache-dir", cls.cache],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
        if cls.completed.returncode != 0:
            raise AssertionError(
                "eval_pointnav subprocess failed\nSTDOUT:\n"
                + cls.completed.stdout[-4000:] + "\nSTDERR:\n"
                + cls.completed.stderr[-4000:])
        with open(os.path.join(cls.out, "run.json"), "r", encoding="utf-8") as handle:
            cls.run_record = json.load(handle)
        with open(os.path.join(cls.out, "episodes.jsonl"), "r",
                  encoding="utf-8") as handle:
            cls.records = [json.loads(line) for line in handle if line.strip()]
        with open(os.path.join(cls.out, "scenes.json"), "r", encoding="utf-8") as handle:
            cls.scene_geometry = json.load(handle)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_policy_and_oracle_are_reported_separately(self):
        self.assertEqual(self.run_record["policy_name"], "path")
        self.assertEqual(self.run_record["summary_scope"], "policy")
        self.assertEqual(self.run_record["policy"]["name"], "path")
        self.assertEqual(self.run_record["oracle"]["name"], "oracle")
        self.assertIn("not a policy", self.run_record["oracle"]["note"])
        self.assertEqual(self.run_record["policy"]["summary"]["episodes"],
                         len(self.records))
        self.assertEqual(self.run_record["oracle"]["summary"]["episodes"],
                         len(self.records))
        self.assertEqual(self.run_record["summary"],
                         self.run_record["policy"]["summary"])
        for scene in self.run_record["scenes"]:
            self.assertIn("oracle_summary", scene)
            self.assertTrue(all("oracle" in result for result in scene["results"]))

    def test_episodes_jsonl_matches_the_replay_contract(self):
        self.assertEqual(len(self.records), 2)
        required = {
            "scene_id", "episode_id", "seed", "start", "goal",
            "start_floor", "goal_floor", "reference_path", "trajectory",
            "success", "spl", "collisions", "timeout", "failure_reason",
        }
        actions = {"left", "right", "forward", "stop"}
        for record in self.records:
            self.assertTrue(required.issubset(record))
            self.assertEqual(record["policy"], "path")
            self.assertEqual(record["yaw_unit"], "degrees")
            self.assertEqual(record["yaw_convention"], "manifest")
            self.assertGreaterEqual(len(record["reference_path"]["points"]), 2)
            self.assertGreaterEqual(len(record["trajectory"]), 2)
            self.assertEqual(record["trajectory"][0]["action"], "start")
            for step in record["trajectory"][1:]:
                self.assertIn(step["action"], actions)
                self.assertEqual(len(step["position"]), 3)
                self.assertIsInstance(step["yaw"], float)
                self.assertIsInstance(step["collision"], bool)

    def test_scenes_json_provides_the_replay_geometry(self):
        self.assertEqual(self.scene_geometry["coordinate_system"], "gltf")
        self.assertEqual(len(self.scene_geometry["scenes"]), 1)
        scene = self.scene_geometry["scenes"][0]
        self.assertEqual(scene["scene_id"], "bench_scene_0000")
        self.assertTrue(scene["rooms"])
        self.assertTrue(scene["floors"])
        for room in scene["rooms"]:
            self.assertIn("floor_id", room)
            self.assertGreaterEqual(len(room["polygon"]), 3)

    def test_repro_check_covers_every_episode(self):
        self.assertTrue(self.run_record["repro_check"])
        for scene in self.run_record["scenes"]:
            self.assertEqual(scene["policy_reproducible_episodes"],
                             scene["episode_count"])
            self.assertEqual(scene["oracle_reproducible_episodes"],
                             scene["episode_count"])


if __name__ == "__main__":
    unittest.main()
