"""Discrete motion / observation tests for PointNavEnv using a fake NavMesh."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from roomgen_nav.env import PointNavEnv, script_actions  # noqa: E402
from roomgen_nav.episodes import Episode  # noqa: E402
from roomgen_nav.pathfinding import CompositePath  # noqa: E402
from roomgen_nav.profiles import HUMANOID  # noqa: E402


class FakeFloor:
    """A rectangular floor with an optional wall at ``wall_x``."""

    def __init__(self, wall_x=None, half=2.0, elevation=0.0):
        self.wall_x = wall_x
        self.half = half
        self.elevation = elevation

    def snap_point(self, point):
        x = min(max(float(point[0]), -self.half), self.half)
        z = min(max(float(point[2]), -self.half), self.half)
        return (x, self.elevation, z)

    def try_step(self, start, end):
        x = min(max(float(end[0]), -self.half), self.half)
        z = min(max(float(end[2]), -self.half), self.half)
        if self.wall_x is not None:
            if float(start[0]) < self.wall_x <= x:
                x = self.wall_x - 1e-3
            elif float(start[0]) > self.wall_x >= x:
                x = self.wall_x + 1e-3
        return (x, self.elevation, z)

    def find_path(self, start, end):
        from roomgen_nav.pathfinding import FloorPath
        distance = ((float(end[0]) - float(start[0])) ** 2
                    + (float(end[2]) - float(start[2])) ** 2) ** 0.5
        return FloorPath([tuple(start), tuple(end)], distance)


class FakeNav:
    def __init__(self, wall_x=None):
        self.entry = SimpleNamespace(scene_id="fake_scene")
        self.floor_nav = FakeFloor(wall_x=wall_x)
        self.floors = {"F0": self.floor_nav}
        self.valid_links = []
        self.links = []

    def floor(self, floor_id):
        return self.floors[floor_id]

    def snap(self, floor_id, point, tolerance=1.0):
        return self.floor_nav.snap_point(point)


def _episode(start, goal, yaw=0.0, start_floor="F0", goal_floor="F0",
             shortest=1.0):
    return Episode(
        scene_id="fake_scene", episode_id=0, seed=7,
        start_position=[float(v) for v in start],
        start_yaw_deg=float(yaw),
        goal_position=[float(v) for v in goal],
        start_floor=start_floor, goal_floor=goal_floor,
        start_room="a", goal_room="b", shortest_distance=float(shortest),
    )


def _env(wall_x=None, **kwargs):
    return PointNavEnv("fake_scene", None, HUMANOID, FakeNav(wall_x=wall_x), **kwargs)


class MotionTest(unittest.TestCase):
    def test_forward_and_turn(self):
        env = _env()
        env.reset("fake_scene", 0, 7, episode=_episode((0, 0, 0), (2, 0, 0)))
        env.step("forward")
        self.assertAlmostEqual(env.position[0], HUMANOID.step_size, places=5)
        self.assertAlmostEqual(env.position[2], 0.0, places=5)
        env.step("left")
        self.assertAlmostEqual(env.yaw_deg, HUMANOID.turn_deg, places=5)
        env.step("right")
        self.assertAlmostEqual(env.yaw_deg, 0.0, places=5)
        obs = env.observation()
        self.assertEqual(obs["step"], 3)
        self.assertIn("rotation", obs)
        self.assertEqual(len(obs["rotation"]), 4)

    def test_success_on_proximity(self):
        env = _env()
        env.reset("fake_scene", 0, 7, episode=_episode((0, 0, 0), (0.9, 0, 0),
                                                       shortest=0.9))
        env.step("forward")
        self.assertFalse(env.done)
        env.step("forward")
        self.assertTrue(env.success)
        self.assertTrue(env.done)
        result = env.evaluate()
        self.assertTrue(result["success"])
        self.assertGreater(result["spl"], 0.0)
        self.assertEqual(result["failure_reason"], None)

    def test_stop_short_is_a_failure(self):
        env = _env()
        env.reset("fake_scene", 0, 7, episode=_episode((0, 0, 0), (1.8, 0, 0)))
        env.step("forward")
        env.step("stop")
        result = env.evaluate()
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reason"], "stopped_short")

    def test_timeout(self):
        env = _env(max_steps=2)
        env.reset("fake_scene", 0, 7, episode=_episode((0, 0, 0), (1.8, 0, 0)))
        env.step("forward")
        env.step("forward")
        result = env.evaluate()
        self.assertFalse(result["success"])
        self.assertTrue(result["timeout"])
        self.assertEqual(result["failure_reason"], "timeout")

    def test_wall_collision_and_stuck(self):
        env = _env(wall_x=0.6, max_steps=100)
        env.reset("fake_scene", 0, 7, episode=_episode((0, 0, 0), (1.8, 0, 0)))
        for _ in range(20):
            env.step("forward")
        self.assertGreaterEqual(env.collisions, 5)
        self.assertTrue(env.done)
        self.assertEqual(env.failure_reason, "stuck")

    def test_invalid_action(self):
        env = _env()
        env.reset("fake_scene", 0, 7, episode=_episode((0, 0, 0), (1.8, 0, 0)))
        env.step("teleport")
        self.assertTrue(env.done)
        self.assertEqual(env.failure_reason, "invalid_action")

    def test_reset_is_reproducible(self):
        env = _env()
        episode = _episode((0.1, 0, -0.2), (1.0, 0, 0.3), yaw=30.0)
        first = env.reset("fake_scene", 0, 7, episode=episode)
        env.step("forward")
        second = env.reset("fake_scene", 0, 7, episode=episode)
        self.assertEqual(first["position"], second["position"])
        self.assertEqual(first["heading_deg"], second["heading_deg"])
        self.assertEqual(second["step"], 0)

    def test_observation_provider_merges_rgbd(self):
        env = _env()
        env.reset("fake_scene", 0, 7, episode=_episode((0, 0, 0), (1.0, 0, 0)))
        env.set_observation_provider(lambda current: {"rgb": [[1, 2, 3]],
                                                      "depth": [[4.0]]})
        obs = env.observation()
        self.assertEqual(obs["rgb"], [[1, 2, 3]])
        self.assertEqual(obs["depth"], [[4.0]])
        self.assertEqual(obs["sensor"]["provider"], "external")

    def test_attach_catalog_makes_reset_self_contained(self):
        env = _env()
        episode = _episode((0.2, 0, 0.1), (1.0, 0, 0.1))
        env.attach_catalog(SimpleNamespace(get=lambda episode_id, seed: episode))
        obs = env.reset("fake_scene", 0, 7)
        self.assertEqual(obs["episode_id"], 0)
        self.assertEqual(obs["position"], episode.start_position)

    def test_scripted_actions_replay(self):
        env = _env()
        episode = _episode((0, 0, 0), (1.5, 0, 0), shortest=1.5)
        env.reset("fake_scene", 0, 7, episode=episode)
        path = CompositePath(points=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
                             distance=1.5)
        actions = script_actions(env, path)
        self.assertIn("forward", actions)
        self.assertEqual(actions[-1], "stop")
        self.assertTrue(env.success)
        self.assertTrue(env.done)
        final = env.observation()
        env.reset("fake_scene", 0, 7, episode=episode)
        env.rollout(actions)
        self.assertTrue(env.success)
        self.assertAlmostEqual(env.position[0], final["position"][0], places=6)
        self.assertAlmostEqual(env.position[2], final["position"][2], places=6)


if __name__ == "__main__":
    unittest.main()
