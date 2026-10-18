"""Agent-profile unit tests: the two shipped tiers and their reachability rules."""

from __future__ import annotations

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from roomgen_nav import profiles as profiles_mod  # noqa: E402
from roomgen_nav.profiles import AgentProfile, HUMANOID, WHEELED  # noqa: E402


class ProfileTest(unittest.TestCase):
    def test_two_distinct_tiers(self):
        self.assertNotEqual(HUMANOID.fingerprint(), WHEELED.fingerprint())
        self.assertGreater(HUMANOID.max_step, WHEELED.max_step)
        self.assertGreater(HUMANOID.max_slope_deg, WHEELED.max_slope_deg)
        self.assertLess(HUMANOID.radius, WHEELED.radius)
        self.assertEqual(HUMANOID.locomotion, "humanoid")
        self.assertEqual(WHEELED.locomotion, "wheeled")

    def test_stair_riser_rule(self):
        riser = 0.19  # the steepest legal generated riser
        self.assertTrue(HUMANOID.can_climb_riser(riser))
        self.assertFalse(WHEELED.can_climb_riser(riser))
        self.assertEqual(HUMANOID.navmesh_settings()["agent_max_climb"],
                         HUMANOID.max_step)

    def test_ramp_slope_rule(self):
        # 0.45 m rise over 2.4 m run = 0.1875 (18.75 %), the split-level demo.
        self.assertTrue(HUMANOID.can_traverse_slope(0.45, 2.4))
        self.assertFalse(WHEELED.can_traverse_slope(0.45, 2.4))
        # 1:12 accessible ramp is fine for the wheeled profile.
        self.assertTrue(WHEELED.can_traverse_slope(1.0, 12.0))
        self.assertAlmostEqual(WHEELED.max_slope, 0.087489, places=5)

    def test_navmesh_settings_are_derived(self):
        settings = WHEELED.navmesh_settings()
        self.assertEqual(set(settings), {"agent_radius", "agent_height",
                                         "agent_max_climb", "agent_max_slope",
                                         "cell_size", "cell_height"})
        self.assertEqual(settings["agent_radius"], WHEELED.radius)
        self.assertEqual(settings["agent_height"], WHEELED.height)
        self.assertEqual(settings["agent_max_climb"], WHEELED.max_step)
        self.assertEqual(settings["agent_max_slope"], WHEELED.max_slope_deg)

    def test_fingerprint_is_stable_and_sensitive(self):
        a = AgentProfile(name="x", radius=0.2, height=1.6, max_step=0.2,
                         max_slope_deg=40.0)
        b = AgentProfile(name="x", radius=0.2, height=1.6, max_step=0.2,
                         max_slope_deg=40.0)
        c = AgentProfile(name="x", radius=0.21, height=1.6, max_step=0.2,
                         max_slope_deg=40.0)
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertNotEqual(a.fingerprint(), c.fingerprint())

    def test_motion_changes_do_not_rebake_the_navmesh(self):
        walk = AgentProfile(name="x", radius=0.2, height=1.6, max_step=0.2,
                            max_slope_deg=40.0, step_size=0.25)
        fast = AgentProfile(name="x", radius=0.2, height=1.6, max_step=0.2,
                            max_slope_deg=40.0, step_size=0.40)
        self.assertEqual(walk.navmesh_fingerprint(), fast.navmesh_fingerprint())
        self.assertNotEqual(walk.fingerprint(), fast.fingerprint())

    def test_invalid_profiles_rejected(self):
        with self.assertRaises(ValueError):
            AgentProfile(name="bad", radius=0.0, height=1.0, max_step=0.1,
                         max_slope_deg=30.0)
        with self.assertRaises(ValueError):
            AgentProfile(name="bad", radius=0.2, height=1.0, max_step=0.1,
                         max_slope_deg=0.0)
        with self.assertRaises(ValueError):
            AgentProfile(name="bad", radius=0.2, height=1.0, max_step=0.1,
                         max_slope_deg=30.0, locomotion="hover")

    def test_get_profile_and_catalog(self):
        self.assertIs(profiles_mod.get_profile("humanoid"), HUMANOID)
        self.assertIn("wheeled", profiles_mod.available_profiles())
        with self.assertRaises(KeyError):
            profiles_mod.get_profile("does_not_exist")

    def test_profile_json_roundtrip(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "profile.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(HUMANOID.to_dict(), handle)
            loaded = profiles_mod.get_profile(path)
            self.assertEqual(loaded.fingerprint(), HUMANOID.fingerprint())
            self.assertEqual(loaded.navmesh_fingerprint(),
                             HUMANOID.navmesh_fingerprint())


if __name__ == "__main__":
    unittest.main()
