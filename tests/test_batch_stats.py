"""Batch accounting: stage grading, exit codes, run ids, retry and resume.

Run::

    python tests/test_batch_stats.py            # or: python tests/test_batch_stats.py -v

``scripts/layout_gen_run.sh`` has to grade a scene on four levels
(layout_ok / render_ok / validation_ok / bundle_ok), propagate the failing
stage as its exit code, and let ``scripts/batch_generate.sh`` aggregate those
levels instead of counting "layout only" or memory-skipped scenes as passed.
These tests drive the real scripts end to end: the layout generator and
``checks.py`` actually run, and the Blender pipeline is replaced by a fake
through ``PANOWORLD_PIPELINE_SH`` that writes a ``validation_report.json``,
so no render is needed to exercise the grading and retry paths.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
HOSTPY = sys.executable
LAYOUT_RUN = os.path.join(REPO_ROOT, "scripts", "layout_gen_run.sh")
BATCH = os.path.join(REPO_ROOT, "scripts", "batch_generate.sh")

# The pipeline seam is bash, so the fake is a small bash script around this
# python program.  Its behaviour comes from <outdir>/fake_mode:
#   ok            -> all checks pass (canonical_resolution follows PANOWORLD_RES)
#   depth_fail    -> depth_matches_geometry fails
#   files_missing -> files_present and every export check fail
#   fail_exit     -> exits 3 without writing a report
#   oom_always    -> SIGKILLs itself (exit 137)
#   oom_then_ok   -> SIGKILLs itself on the first call only
FAKE_PIPELINE = """#!/usr/bin/env bash
set -u
SPEC="$1"
OUT="$2"
MODE="ok"
if [[ -f "${OUT}/fake_mode" ]]; then MODE="$(cat "${OUT}/fake_mode")"; fi
CALLS=0
if [[ -f "${OUT}/fake_calls" ]]; then CALLS="$(cat "${OUT}/fake_calls")"; fi
CALLS=$(( CALLS + 1 ))
echo "${CALLS}" > "${OUT}/fake_calls"
if [[ "${MODE}" == "oom_always" ]]; then
  echo "fake pipeline: simulated OOM (call ${CALLS})" >&2
  kill -9 $$
fi
if [[ "${MODE}" == "oom_then_ok" ]] && [[ ${CALLS} -eq 1 ]]; then
  echo "fake pipeline: simulated OOM (call ${CALLS})" >&2
  kill -9 $$
fi
if [[ "${MODE}" == "fail_exit" ]]; then
  echo "fake pipeline: simulated crash" >&2
  exit 3
fi
RES="${PANOWORLD_RES:-1024x512}"
CANONICAL="false"
if [[ "${RES}" == "2048x1024" ]]; then CANONICAL="true"; fi
DEPTH="true"
if [[ "${MODE}" == "depth_fail" ]]; then DEPTH="false"; fi
FILES="true"
if [[ "${MODE}" == "files_missing" ]]; then FILES="false"; fi
if [[ "${FILES}" == "true" ]]; then
  mkdir -p "${OUT}/scene/viewpoints/0000"
  echo "fake panorama" > "${OUT}/scene/viewpoints/0000/panoImage_2048.png"
fi
"@PYTHON@" - "${OUT}" "${CANONICAL}" "${DEPTH}" "${FILES}" <<'PYEOF'
import json, os, sys
out, canonical, depth, files = sys.argv[1:5]
flags = {"canonical": canonical == "true", "files": files == "true", "depth": depth == "true"}
checks = [
    {"name": "canonical_resolution", "pass": flags["canonical"]},
    {"name": "files_present", "pass": flags["files"]},
    {"name": "pano_format", "pass": flags["files"]},
    {"name": "depth_format", "pass": flags["files"]},
    {"name": "extrinsics_valid", "pass": flags["files"]},
    {"name": "camera_pose_matches_spec", "pass": flags["files"]},
    {"name": "depth_matches_geometry", "pass": flags["depth"]},
    {"name": "map_json_reachable", "pass": flags["files"]},
    {"name": "topdown_non_empty", "pass": flags["files"]},
]
report = {"kind": "blender_agent_p0_validation",
          "pass": all(check["pass"] for check in checks), "checks": checks}
with open(os.path.join(out, "validation_report.json"), "w", encoding="utf-8") as handle:
    json.dump(report, handle)
PYEOF
exit 0
"""


def base_env(**overrides):
    """The ambient environment without any PANOWORLD_* leakage."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("PANOWORLD_")}
    env["PANOWORLD_HOSTPY"] = HOSTPY
    for key, value in overrides.items():
        env[key] = str(value)
    return env


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class ScriptCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="batch_stats_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        with open(os.path.join(self.tmp, "fake_pipeline.sh"), "w", encoding="utf-8") as handle:
            handle.write(FAKE_PIPELINE.replace("@PYTHON@", HOSTPY))
        self.fake_pipeline = os.path.join(self.tmp, "fake_pipeline.sh")

    def run_layout(self, seed, out, rooms=None, env=None):
        args = ["bash", LAYOUT_RUN, str(seed), out]
        if rooms is not None:
            args.append(str(rooms))
        return subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT,
                              env=env or base_env())

    def run_batch(self, start, count, root, env=None):
        return subprocess.run(["bash", BATCH, str(start), str(count), root],
                              capture_output=True, text=True, cwd=REPO_ROOT,
                              env=env or base_env())

    def summary_of(self, out):
        return load_json(os.path.join(out, "run_summary.json"))

    def batch_summary_of(self, root):
        return load_json(os.path.join(root, "batch_summary.json"))


class LayoutRunExitCodeTest(ScriptCase):
    def test_layout_only_grades_bundle_ok_and_exit0(self):
        out = os.path.join(self.tmp, "scene")
        result = self.run_layout(0, out, env=base_env(PANOWORLD_SKIP_RENDER=1,
                                                      PANOWORLD_RUN_ID="unit-layout-only"))
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertEqual(summary["run_id"], "unit-layout-only")
        self.assertFalse(summary["render_requested"])
        self.assertEqual(summary["ok"], {"layout_ok": True, "render_ok": None,
                                         "validation_ok": None, "bundle_ok": True})
        self.assertEqual(summary["stages"]["render"], "skipped_by_request")
        self.assertEqual(summary["stages"]["validation"], "not_run")
        self.assertEqual(summary["exit_code"], 0)
        self.assertIsNone(summary["failure"])
        self.assertTrue(summary["checks"]["pass"])

    def test_bad_rooms_fails_layout_with_exit2(self):
        out = os.path.join(self.tmp, "bad")
        # 0 rooms is outside layout_gen's range whatever that range grows to
        result = self.run_layout(5, out, rooms=0,
                                 env=base_env(PANOWORLD_SKIP_RENDER=1, PANOWORLD_RUN_ID="unit-bad"))
        self.assertEqual(result.returncode, 2, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertFalse(summary["ok"]["layout_ok"])
        self.assertFalse(summary["ok"]["bundle_ok"])
        self.assertEqual(summary["stages"]["layout"], "failed")
        self.assertEqual(summary["stages"]["checks"], "not_run")
        self.assertEqual(summary["failure"]["stage"], "layout")
        self.assertEqual(summary["failure"]["kind"], "generation_failed")
        self.assertEqual(summary["exit_code"], 2)
        self.assertFalse(os.path.isfile(os.path.join(out, "spec.json")))

    def test_stale_reports_are_dropped_and_run_id_recorded(self):
        out = os.path.join(self.tmp, "stale")
        os.makedirs(out)
        with open(os.path.join(out, "spec.json"), "w", encoding="utf-8") as handle:
            handle.write('{"stale": true}')
        with open(os.path.join(out, "validation_report.json"), "w", encoding="utf-8") as handle:
            handle.write('{"pass": true, "checks": [{"name": "files_present", "pass": true}]}')
        with open(os.path.join(out, "run_summary.json"), "w", encoding="utf-8") as handle:
            handle.write('{"ok": {"bundle_ok": true}, "run_id": "old-run"}')
        result = self.run_layout(1, out, env=base_env(PANOWORLD_SKIP_RENDER=1,
                                                      PANOWORLD_RUN_ID="unit-stale"))
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertEqual(summary["run_id"], "unit-stale")
        self.assertIsNone(summary["failure"])
        # the stale render report cannot survive a run that did not render
        self.assertFalse(os.path.isfile(os.path.join(out, "validation_report.json")))
        spec = load_json(os.path.join(out, "spec.json"))
        self.assertIn("rooms", spec)

    def test_render_success_grades_all_levels(self):
        out = os.path.join(self.tmp, "render_ok")
        env = base_env(PANOWORLD_RES="1024x512", PANOWORLD_SAMPLES=8,
                       PANOWORLD_MIN_FREE_MB=1, PANOWORLD_PIPELINE_SH=self.fake_pipeline,
                       PANOWORLD_RUN_ID="unit-render-ok")
        result = self.run_layout(2, out, env=env)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertEqual(summary["ok"], {"layout_ok": True, "render_ok": True,
                                         "validation_ok": True, "bundle_ok": True})
        self.assertEqual(summary["stages"]["render"], "ok")
        self.assertEqual(summary["stages"]["validation"], "ok")
        self.assertEqual(len(summary["render"]["attempts"]), 1)
        self.assertEqual(summary["render"]["attempts"][0]["resolution"], "1024x512")
        # non-canonical resolution: only canonical_resolution may fail
        self.assertTrue(summary["validation"]["pass_at_resolution"])
        self.assertEqual(summary["validation"]["failed"], ["canonical_resolution"])
        self.assertEqual(summary["exit_code"], 0)

    def test_pipeline_failure_exits_4(self):
        out = os.path.join(self.tmp, "render_fail")
        os.makedirs(out)
        with open(os.path.join(out, "fake_mode"), "w", encoding="utf-8") as handle:
            handle.write("fail_exit")
        env = base_env(PANOWORLD_RES="1024x512", PANOWORLD_SAMPLES=8,
                       PANOWORLD_MIN_FREE_MB=1, PANOWORLD_PIPELINE_SH=self.fake_pipeline,
                       PANOWORLD_RUN_ID="unit-render-fail")
        result = self.run_layout(3, out, env=env)
        self.assertEqual(result.returncode, 4, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertTrue(summary["ok"]["layout_ok"])
        self.assertFalse(summary["ok"]["render_ok"])
        self.assertIsNone(summary["ok"]["validation_ok"])
        self.assertEqual(summary["stages"]["render"], "failed")
        self.assertEqual(summary["failure"]["stage"], "render")
        self.assertEqual(summary["failure"]["kind"], "pipeline_exit_3")
        self.assertEqual(summary["render"]["pipeline_exit_code"], 3)

    def test_validation_failure_exits_5(self):
        out = os.path.join(self.tmp, "validation_fail")
        os.makedirs(out)
        with open(os.path.join(out, "fake_mode"), "w", encoding="utf-8") as handle:
            handle.write("depth_fail")
        env = base_env(PANOWORLD_RES="1024x512", PANOWORLD_SAMPLES=8,
                       PANOWORLD_MIN_FREE_MB=1, PANOWORLD_PIPELINE_SH=self.fake_pipeline,
                       PANOWORLD_RUN_ID="unit-validation-fail")
        result = self.run_layout(4, out, env=env)
        self.assertEqual(result.returncode, 5, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertTrue(summary["ok"]["render_ok"], "the render artifacts were complete")
        self.assertFalse(summary["ok"]["validation_ok"])
        self.assertEqual(summary["failure"]["stage"], "validation")
        self.assertEqual(summary["failure"]["kind"], "validation_failed")
        self.assertEqual(summary["failure"]["checks"], ["canonical_resolution",
                                                        "depth_matches_geometry"])

    def test_missing_viewpoints_fail_the_render_exit4(self):
        out = os.path.join(self.tmp, "files_missing")
        os.makedirs(out)
        with open(os.path.join(out, "fake_mode"), "w", encoding="utf-8") as handle:
            handle.write("files_missing")
        env = base_env(PANOWORLD_RES="1024x512", PANOWORLD_SAMPLES=8,
                       PANOWORLD_MIN_FREE_MB=1, PANOWORLD_PIPELINE_SH=self.fake_pipeline,
                       PANOWORLD_RUN_ID="unit-files-missing")
        result = self.run_layout(5, out, env=env)
        self.assertEqual(result.returncode, 4, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertFalse(summary["ok"]["render_ok"])
        self.assertEqual(summary["failure"]["stage"], "render")
        self.assertEqual(summary["failure"]["kind"], "artifacts_missing")

    def test_memory_oom_is_retried_once_with_a_reduced_profile(self):
        out = os.path.join(self.tmp, "oom_retry")
        os.makedirs(out)
        with open(os.path.join(out, "fake_mode"), "w", encoding="utf-8") as handle:
            handle.write("oom_then_ok")
        env = base_env(PANOWORLD_RES="1024x512", PANOWORLD_SAMPLES=24,
                       PANOWORLD_MIN_FREE_MB=1, PANOWORLD_PIPELINE_SH=self.fake_pipeline,
                       PANOWORLD_RUN_ID="unit-oom-retry")
        result = self.run_layout(6, out, env=env)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertEqual(summary["ok"]["bundle_ok"], True)
        self.assertTrue(summary["render"]["retry_used"])
        attempts = summary["render"]["attempts"]
        self.assertEqual([attempt["status"] for attempt in attempts],
                         ["killed_signal_9", "ok"])
        self.assertEqual(attempts[0]["resolution"], "1024x512")
        self.assertEqual(attempts[1]["resolution"], "512x256")
        self.assertEqual(attempts[1]["samples"], 16)

    def test_memory_oom_after_the_retry_is_pending_retry(self):
        out = os.path.join(self.tmp, "oom_always")
        os.makedirs(out)
        with open(os.path.join(out, "fake_mode"), "w", encoding="utf-8") as handle:
            handle.write("oom_always")
        env = base_env(PANOWORLD_RES="1024x512", PANOWORLD_SAMPLES=24,
                       PANOWORLD_MIN_FREE_MB=1, PANOWORLD_PIPELINE_SH=self.fake_pipeline,
                       PANOWORLD_RUN_ID="unit-oom-always")
        result = self.run_layout(7, out, env=env)
        self.assertEqual(result.returncode, 4, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertEqual(summary["stages"]["render"], "pending_retry")
        self.assertFalse(summary["ok"]["render_ok"])
        self.assertFalse(summary["ok"]["bundle_ok"])
        self.assertIsNone(summary["ok"]["validation_ok"])
        self.assertEqual(summary["failure"]["stage"], "render")
        self.assertEqual(summary["failure"]["kind"], "killed_signal_9")
        self.assertEqual(len(summary["render"]["attempts"]), 2)

    def test_memory_skip_without_retry_is_pending_retry_not_pass(self):
        out = os.path.join(self.tmp, "mem_skip")
        env = base_env(PANOWORLD_MIN_FREE_MB=99999999, PANOWORLD_RETRY_MEMORY=0,
                       PANOWORLD_RUN_ID="unit-mem-skip")
        result = self.run_layout(8, out, env=env)
        self.assertEqual(result.returncode, 4, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertEqual(summary["stages"]["render"], "pending_retry")
        self.assertFalse(summary["ok"]["bundle_ok"])
        self.assertEqual(summary["failure"]["kind"], "skipped_low_memory")
        self.assertEqual([attempt["status"] for attempt in summary["render"]["attempts"]],
                         ["skipped_low_memory"])


class BatchSummaryTest(ScriptCase):
    def test_layout_only_batch_levels_run_ids_and_exit0(self):
        root = os.path.join(self.tmp, "layout_batch")
        result = self.run_batch(0, 2, root)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = self.batch_summary_of(root)
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["levels"],
                         {"layout_ok": 2, "render_ok": None, "validation_ok": None, "bundle_ok": 2})
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["pending_retry"], 0)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["failure_reasons"], {})
        self.assertFalse(summary["render_requested"])
        for row in summary["scenes"]:
            self.assertEqual(row["run_id"], f"{summary['batch_id']}-seed{row['seed']}")
            self.assertEqual(row["status"], "bundle_ok")
            self.assertFalse(row["resumed"])

    def test_render_batch_grades_levels_and_failure_distribution(self):
        root = os.path.join(self.tmp, "render_batch")
        os.makedirs(os.path.join(root, "scene_0001"))
        with open(os.path.join(root, "scene_0001", "fake_mode"), "w", encoding="utf-8") as handle:
            handle.write("depth_fail")
        env = base_env(PANOWORLD_BATCH_RENDER=1, PANOWORLD_RES="1024x512",
                       PANOWORLD_SAMPLES=8, PANOWORLD_MIN_FREE_MB=1,
                       PANOWORLD_PIPELINE_SH=self.fake_pipeline)
        result = self.run_batch(0, 3, root, env=env)
        self.assertEqual(result.returncode, 1, result.stderr[-2000:])
        summary = self.batch_summary_of(root)
        self.assertTrue(summary["render_requested"])
        self.assertEqual(summary["levels"],
                         {"layout_ok": 3, "render_ok": 3, "validation_ok": 2, "bundle_ok": 2})
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["failure_reasons"], {"validation:validation_failed": 1})
        self.assertEqual(summary["failed_checks"], {"canonical_resolution": 1,
                                                    "depth_matches_geometry": 1})
        # the summary numbers must match the artifacts, not just the report
        for row in summary["scenes"]:
            scene_dir = os.path.join(root, row["dir"])
            report = load_json(os.path.join(scene_dir, "validation_report.json"))
            failed = [check["name"] for check in report["checks"] if not check["pass"]]
            expected_validation = report["pass"] or failed == ["canonical_resolution"]
            self.assertEqual(row["ok"]["validation_ok"], expected_validation)
            self.assertEqual(row["ok"]["bundle_ok"],
                             row["ok"]["layout_ok"] and row["ok"]["render_ok"]
                             and row["ok"]["validation_ok"])
        statuses = {row["seed"]: row["status"] for row in summary["scenes"]}
        self.assertEqual(statuses, {0: "bundle_ok", 1: "failed", 2: "bundle_ok"})

    def test_render_batch_memory_pending_is_not_passed(self):
        root = os.path.join(self.tmp, "pending_batch")
        env = base_env(PANOWORLD_BATCH_RENDER=1, PANOWORLD_MIN_FREE_MB=99999999,
                       PANOWORLD_RETRY_MEMORY=0, PANOWORLD_PIPELINE_SH=self.fake_pipeline)
        result = self.run_batch(0, 2, root, env=env)
        self.assertEqual(result.returncode, 1, result.stderr[-2000:])
        summary = self.batch_summary_of(root)
        self.assertEqual(summary["levels"]["bundle_ok"], 0)
        self.assertEqual(summary["levels"]["render_ok"], 0)
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["pending_retry"], 2)
        for row in summary["scenes"]:
            self.assertEqual(row["status"], "pending_retry")
            self.assertEqual(row["failure"]["kind"], "skipped_low_memory")

    def test_resume_keeps_completed_seeds_and_their_run_ids(self):
        root = os.path.join(self.tmp, "resume_batch")
        first = self.run_batch(0, 2, root)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        run_ids = {row["seed"]: row["run_id"] for row in self.batch_summary_of(root)["scenes"]}
        second = self.run_batch(0, 2, root, env=base_env(PANOWORLD_BATCH_RESUME=1))
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        summary = self.batch_summary_of(root)
        self.assertEqual(summary["levels"]["bundle_ok"], 2)
        for row in summary["scenes"]:
            self.assertTrue(row["resumed"], f"seed {row['seed']} was re-run")
            self.assertEqual(row["run_id"], run_ids[row["seed"]])
        # a resume without --resume must re-run and therefore refresh the run ids
        third = self.run_batch(0, 2, root)
        self.assertEqual(third.returncode, 0, third.stderr[-2000:])
        for row in self.batch_summary_of(root)["scenes"]:
            self.assertFalse(row["resumed"])
            self.assertNotEqual(row["run_id"], run_ids[row["seed"]])

    def test_resume_recomputes_when_the_requested_mode_changes(self):
        root = os.path.join(self.tmp, "mode_change")
        first = self.run_batch(0, 2, root)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        old_ids = {row["seed"]: row["run_id"] for row in self.batch_summary_of(root)["scenes"]}
        env = base_env(PANOWORLD_BATCH_RENDER=1, PANOWORLD_BATCH_RESUME=1,
                       PANOWORLD_RES="1024x512", PANOWORLD_SAMPLES=8,
                       PANOWORLD_MIN_FREE_MB=1, PANOWORLD_PIPELINE_SH=self.fake_pipeline)
        second = self.run_batch(0, 2, root, env=env)
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        summary = self.batch_summary_of(root)
        self.assertEqual(summary["levels"]["bundle_ok"], 2)
        for row in summary["scenes"]:
            self.assertFalse(row["resumed"], "a layout-only bundle cannot satisfy a render batch")
            self.assertNotEqual(row["run_id"], old_ids[row["seed"]])
            self.assertTrue(row["ok"]["render_ok"])

    def test_summary_only_rebuilds_without_running_anything(self):
        root = os.path.join(self.tmp, "summary_only")
        first = self.run_batch(0, 2, root)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        run_ids = {row["seed"]: row["run_id"] for row in self.batch_summary_of(root)["scenes"]}
        log = os.path.join(root, "scene_0000", "layout_gen.log")
        self.assertTrue(os.path.isfile(log))
        os.remove(log)
        second = self.run_batch(0, 2, root, env=base_env(PANOWORLD_BATCH_SUMMARY_ONLY=1))
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        summary = self.batch_summary_of(root)
        self.assertTrue(summary["summary_only"])
        self.assertEqual(summary["levels"]["bundle_ok"], 2)
        for row in summary["scenes"]:
            self.assertEqual(row["run_id"], run_ids[row["seed"]])
        self.assertFalse(os.path.isfile(log), "summary-only mode must not run layout_gen")

    def test_stale_summary_is_not_trusted_without_resume(self):
        root = os.path.join(self.tmp, "stale_batch")
        scene = os.path.join(root, "scene_0000")
        os.makedirs(scene)
        with open(os.path.join(scene, "spec.json"), "w", encoding="utf-8") as handle:
            handle.write('{"stale": true}')
        with open(os.path.join(scene, "run_summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"kind": "layout_gen_run", "run_id": "stale-run",
                       "render_requested": False,
                       "ok": {"layout_ok": True, "render_ok": None,
                              "validation_ok": None, "bundle_ok": True}}, handle)
        result = self.run_batch(0, 1, root)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = self.batch_summary_of(root)
        row = summary["scenes"][0]
        self.assertFalse(row["resumed"])
        self.assertEqual(row["run_id"], f"{summary['batch_id']}-seed0")
        self.assertNotEqual(row["run_id"], "stale-run")

    def test_old_summary_format_is_still_readable(self):
        root = os.path.join(self.tmp, "legacy_batch")
        scene = os.path.join(root, "scene_0000")
        os.makedirs(os.path.join(scene, "scene", "viewpoints", "0000"))
        with open(os.path.join(scene, "spec.json"), "w", encoding="utf-8") as handle:
            json.dump({"rooms": []}, handle)
        with open(os.path.join(scene, "checks_report.json"), "w", encoding="utf-8") as handle:
            json.dump({"pass": True, "checks": [], "violations": []}, handle)
        with open(os.path.join(scene, "validation_report.json"), "w", encoding="utf-8") as handle:
            json.dump({"pass": False, "checks": [{"name": "canonical_resolution", "pass": False}]}, handle)
        with open(os.path.join(scene, "scene", "viewpoints", "0000", "panoImage_2048.png"), "w") as handle:
            handle.write("x")
        with open(os.path.join(scene, "run_summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"kind": "layout_gen_run", "seed": 0,
                       "render": {"status": "validation_flagged", "resolution": "1024x512",
                                  "samples": 24, "pipeline_exit_code": 1},
                       "checks": {"pass": True},
                       "validation": {"pass": False, "pass_at_resolution": True,
                                      "canonical_resolution": False,
                                      "failed": ["canonical_resolution"]},
                       "rooms": ["bedroom"], "cameras": ["0000"], "furniture": 3}, handle)
        result = self.run_batch(0, 1, root, env=base_env(PANOWORLD_BATCH_RENDER=1,
                                                         PANOWORLD_BATCH_SUMMARY_ONLY=1))
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = self.batch_summary_of(root)
        self.assertEqual(summary["levels"],
                         {"layout_ok": 1, "render_ok": 1, "validation_ok": 1, "bundle_ok": 1})
        row = summary["scenes"][0]
        self.assertTrue(row["legacy_summary"])
        self.assertEqual(row["status"], "bundle_ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
