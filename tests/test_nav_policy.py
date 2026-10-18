"""Closed-loop ShortestPathPolicy, JSONL contract and policy/oracle metric tests.

The host interpreter has no Habitat-Sim, so these tests use the same tiny fake
NavMesh as ``test_nav_env``.  They pin the acceptance properties that do not
need a bake: the policy reads waypoints instead of stored actions, emits only
the discrete action vocabulary, stops on arrival, records collisions, is
deterministic, and the summary keeps policy and oracle numbers separate.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from roomgen_nav import metrics  # noqa: E402
from roomgen_nav.env import (  # noqa: E402
    ACTIONS,
    PointNavEnv,
    ShortestPathPolicy,
    run_action_sequence,
)
from roomgen_nav.episodes import Episode, episode_record, write_episode_jsonl  # noqa: E402
from roomgen_nav.profiles import HUMANOID  # noqa: E402


class FakeFloor:
    """A rectangular floor with an optional blocking wall at ``wall_x``."""

    def __init__(self, wall_x=None, half=6.0, elevation=0.0):
        self.wall_x = wall_x
        self.half = half
        self.elevation = elevation

    def snap_point(self, point):
        return (min(max(float(point[0]), -self.half), self.half),
                self.elevation,
                min(max(float(point[2]), -self.half), self.half))

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


def make_episode(start, goal, yaw=0.0, shortest=None, path=None,
                 actions=None, start_floor="F0", goal_floor="F0"):
    return Episode(
        scene_id="fake_scene", episode_id=0, seed=7,
        start_position=[float(value) for value in start],
        start_yaw_deg=float(yaw),
        goal_position=[float(value) for value in goal],
        start_floor=start_floor, goal_floor=goal_floor,
        start_room="a", goal_room="b",
        shortest_distance=float(shortest if shortest is not None else
                                ((goal[0] - start[0]) ** 2
                                 + (goal[2] - start[2]) ** 2) ** 0.5),
        reference_path=[list(point) for point in (path or [start, goal])],
        actions=list(actions or []),
    )


def make_env(wall_x=None, profile=HUMANOID, auto_success=False, **kwargs):
    return PointNavEnv("fake_scene", None, profile, FakeNav(wall_x=wall_x),
                       auto_success=auto_success, **kwargs)


class PolicyMotionTest(unittest.TestCase):
    def test_straight_path_succeeds_and_finishes_with_stop(self):
        env = make_env()
        episode = make_episode((0, 0, 0), (2, 0, 0),
                               path=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        run = ShortestPathPolicy(env).run(episode)
        self.assertTrue(run["result"]["success"])
        self.assertEqual(run["result"]["failure_reason"], None)
        self.assertEqual(run["actions"][-1], "stop")
        self.assertTrue(set(run["actions"]).issubset(set(ACTIONS)))
        self.assertEqual(run["result"]["collisions"], 0)
        self.assertLessEqual(run["result"]["final_distance"], HUMANOID.success_radius)

    def test_policy_ignores_the_stored_oracle_actions(self):
        env = make_env()
        # Deliberately nonsense oracle actions: a replay would walk backwards
        # into the wall of its budget and fail.
        episode = make_episode((0, 0, 0), (2, 0, 0),
                               path=[[0, 0, 0], [2, 0, 0]],
                               actions=["left", "left", "right", "stop"])
        run = ShortestPathPolicy(env).run(episode)
        self.assertTrue(run["result"]["success"])
        self.assertNotEqual(run["actions"], episode.actions)
        self.assertIn("forward", run["actions"])

    def test_turn_direction_uses_profile_turn_angle(self):
        from roomgen_nav.profiles import AgentProfile
        profile = AgentProfile(name="turny", radius=0.2, height=1.5,
                               max_step=0.2, max_slope_deg=40.0,
                               step_size=0.5, turn_deg=30.0,
                               max_steps=100, success_radius=0.5)
        env = make_env(profile=profile)
        episode = make_episode((0, 0, 0), (2, 0, 0), yaw=90.0)
        policy = ShortestPathPolicy(env)
        self.assertEqual(policy.step_size, 0.5)
        self.assertEqual(policy.turn_deg, 30.0)
        self.assertAlmostEqual(policy.align_deg, 22.5)
        run = policy.run(episode)
        self.assertTrue(run["result"]["success"])
        # Facing -Z, the goal lies to the right: three 30-degree right turns.
        self.assertEqual(run["actions"][:3], ["right", "right", "right"])
        self.assertIn("forward", run["actions"])

    def test_collision_is_recorded_and_fails_with_a_reason(self):
        env = make_env(wall_x=0.5)
        episode = make_episode((0, 0, 0), (4, 0, 0), path=[[0, 0, 0], [4, 0, 0]])
        run = ShortestPathPolicy(env).run(episode)
        self.assertFalse(run["result"]["success"])
        self.assertGreater(run["result"]["collisions"], 0)
        self.assertIn(run["result"]["failure_reason"], ("stuck", "timeout"))
        collisions = [step for step in run["trajectory"] if step["collision"]]
        self.assertTrue(collisions)
        self.assertTrue(all(step["action"] == "forward" for step in collisions))

    def test_stop_when_start_is_already_within_success_radius(self):
        env = make_env()
        episode = make_episode((0, 0, 0), (0.3, 0, 0), path=[[0, 0, 0], [0.3, 0, 0]])
        run = ShortestPathPolicy(env).run(episode)
        self.assertEqual(run["actions"], ["stop"])
        self.assertTrue(run["result"]["success"])
        self.assertEqual(run["result"]["steps"], 1)

    def test_actions_and_positions_are_reproducible(self):
        episode = make_episode((0, 0, 0), (2, 0, 0), yaw=30.0,
                               path=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        first = ShortestPathPolicy(make_env()).run(episode)
        second = ShortestPathPolicy(make_env()).run(episode)
        self.assertEqual(first["actions"], second["actions"])
        self.assertEqual([step["position"] for step in first["trajectory"]],
                         [step["position"] for step in second["trajectory"]])
        self.assertEqual(first["result"]["success"], second["result"]["success"])

    def test_trajectory_schema_and_oracle_runner(self):
        env = make_env()
        episode = make_episode((0, 0, 0), (1.0, 0, 0), actions=["forward", "stop"])
        env.reset("fake_scene", episode.episode_id, episode.seed, episode=episode)
        run = run_action_sequence(env, episode.actions)
        self.assertEqual(run["actions"], ["forward", "stop"])
        self.assertEqual(run["trajectory"][0]["action"], "start")
        for step in run["trajectory"]:
            self.assertEqual(set(step), {"position", "yaw", "action",
                                         "collision", "floor_id"})
            self.assertEqual(len(step["position"]), 3)
            self.assertIsInstance(step["yaw"], float)
            self.assertIsInstance(step["collision"], bool)


class EpisodeRecordTest(unittest.TestCase):
    def test_jsonl_record_matches_replay_contract(self):
        episode = make_episode((0, 0, 0), (2, 0, 0),
                               path=[[0, 0, 0], [2, 0, 0]])
        run = ShortestPathPolicy(make_env()).run(episode)
        record = episode_record(episode, run["result"], run["trajectory"],
                                policy="path", profile=HUMANOID)
        required = {
            "scene_id", "episode_id", "seed", "start", "goal",
            "start_floor", "goal_floor", "reference_path", "trajectory",
            "success", "spl", "collisions", "timeout", "failure_reason",
            "yaw_unit", "yaw_convention",
        }
        self.assertTrue(required.issubset(record))
        self.assertEqual(record["reference_path"]["points"],
                         [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.assertEqual(record["trajectory"][0]["action"], "start")
        self.assertEqual(record["trajectory"][-1]["action"], "stop")
        self.assertTrue(record["success"])
        self.assertEqual(record["yaw_unit"], "degrees")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "episodes.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                write_episode_jsonl(handle, record)
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.loads(handle.readline())
        self.assertEqual(loaded["episode_id"], 0)
        self.assertEqual(len(loaded["trajectory"]), len(run["trajectory"]))


class PolicyMetricsTest(unittest.TestCase):
    def test_policy_and_oracle_summaries_stay_separate(self):
        policy = metrics.summarize_results([
            {"success": True, "spl": 0.8, "collisions": 0, "timeout": False,
             "final_distance": 0.2, "steps": 20, "path_length": 4.0,
             "cross_floor": True, "failure_reason": None},
            {"success": False, "spl": 0.0, "collisions": 4, "timeout": True,
             "final_distance": 3.0, "steps": 100, "path_length": 2.0,
             "cross_floor": True, "failure_reason": "timeout"},
        ])
        oracle = metrics.summarize_results([
            {"success": True, "spl": 1.0, "collisions": 0, "timeout": False,
             "final_distance": 0.1, "steps": 20, "path_length": 4.0,
             "cross_floor": True, "failure_reason": None},
            {"success": True, "spl": 1.0, "collisions": 0, "timeout": False,
             "final_distance": 0.1, "steps": 100, "path_length": 4.0,
             "cross_floor": True, "failure_reason": None},
        ])
        self.assertAlmostEqual(policy["success_rate"], 0.5)
        self.assertAlmostEqual(policy["spl_all"], 0.4)
        self.assertAlmostEqual(oracle["success_rate"], 1.0)
        self.assertEqual(policy["cross_floor"]["episodes"], 2)
        self.assertAlmostEqual(policy["cross_floor"]["success_rate"], 0.5)
        self.assertAlmostEqual(policy["cross_floor"]["spl_all"], 0.4)
        text = metrics.format_summary(policy, heading="POLICY: path")
        self.assertIn("POLICY: path", text)
        self.assertIn("SPL (all)", text)
        self.assertIn("timeout", text)

    def test_failure_reason_distribution(self):
        summary = metrics.summarize_results([
            {"success": False, "failure_reason": "stuck", "timeout": False,
             "collisions": 10, "spl": 0.0, "final_distance": 2.0, "steps": 30},
            {"success": False, "failure_reason": "stuck", "timeout": False,
             "collisions": 10, "spl": 0.0, "final_distance": 2.0, "steps": 30},
            {"success": False, "failure_reason": "timeout", "timeout": True,
             "collisions": 0, "spl": 0.0, "final_distance": 1.0, "steps": 500},
        ])
        self.assertEqual(summary["failure_reasons"], {"stuck": 2, "timeout": 1})
        self.assertAlmostEqual(summary["failure_rate"], 1.0)
        self.assertEqual(summary["timeouts"], 1)


if __name__ == "__main__":
    unittest.main()
