"""Metric aggregation tests (success rate, SPL, collisions, timeout, distance)."""

from __future__ import annotations

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from roomgen_nav import metrics  # noqa: E402


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.results = [
            {"success": True, "spl": 0.75, "collisions": 0, "timeout": False,
             "final_distance": 0.2, "steps": 10, "path_length": 4.0,
             "cross_floor": False, "failure_reason": None},
            {"success": True, "spl": 0.5, "collisions": 2, "timeout": False,
             "final_distance": 0.4, "steps": 20, "path_length": 8.0,
             "cross_floor": True, "failure_reason": None},
            {"success": False, "spl": 0.0, "collisions": 5, "timeout": True,
             "final_distance": 3.0, "steps": 50, "path_length": 1.0,
             "cross_floor": False, "failure_reason": "timeout"},
            {"success": False, "spl": 0.0, "collisions": 0, "timeout": False,
             "final_distance": 1.5, "steps": 12, "path_length": 2.0,
             "cross_floor": True, "failure_reason": "stopped_short"},
        ]

    def test_summary_numbers(self):
        summary = metrics.summarize_results(self.results)
        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(summary["successes"], 2)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["spl"], 0.625)
        self.assertEqual(summary["collisions"], 7)
        self.assertAlmostEqual(summary["collisions_per_episode"], 1.75)
        self.assertEqual(summary["timeouts"], 1)
        self.assertAlmostEqual(summary["timeout_rate"], 0.25)
        self.assertAlmostEqual(summary["final_distance_mean"], 1.275)
        self.assertEqual(summary["failure_reasons"],
                         {"stopped_short": 1, "timeout": 1})
        self.assertEqual(summary["cross_floor"]["episodes"], 2)
        self.assertAlmostEqual(summary["cross_floor"]["success_rate"], 0.5)
        self.assertEqual(summary["same_floor"]["episodes"], 2)
        self.assertAlmostEqual(summary["same_floor"]["success_rate"], 0.5)

    def test_empty_summary_and_format(self):
        summary = metrics.summarize_results([])
        self.assertEqual(summary["episodes"], 0)
        self.assertEqual(summary["success_rate"], 0.0)
        text = metrics.format_summary(summary)
        self.assertIn("episodes", text)
        self.assertIn("success rate", text)

    def test_aggregate_run(self):
        run = {"scenes": [{"scene_id": "a", "results": self.results[:2],
                           "summary": metrics.summarize_results(self.results[:2])},
                          {"scene_id": "b", "results": self.results[2:],
                           "summary": metrics.summarize_results(self.results[2:])}]}
        metrics.aggregate_run(run)
        self.assertEqual(run["summary"]["episodes"], 4)
        self.assertAlmostEqual(run["summary"]["success_rate"], 0.5)
        self.assertIn("a", run["scene_summaries"])
        self.assertIn("b", run["scene_summaries"])


if __name__ == "__main__":
    unittest.main()
