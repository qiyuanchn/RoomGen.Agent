"""Batch accounting: stage grading, exit codes, run ids, retry and resume.

Run::

    python tests/test_batch_stats.py            # or: python tests/test_batch_stats.py -v

``scripts/layout_gen_run.sh`` has to grade a scene on four levels
(layout_ok / render_ok / validation_ok / bundle_ok), propagate the failing
stage as its exit code, and let ``scripts/batch_generate.sh`` aggregate those
levels instead of counting "layout only" or memory-skipped scenes as passed.
These tests drive the real scripts end to end: the layout generator and
``checks.py`` actually run, and the Blender pipeline is replaced by a fake
through ``PANOWORLD_PIPELINE_SH`` that writes a ``validation_report.json``
plus the real artifact files (panorama at the requested resolution, uint16
depth map, camera pose, map, topdown), so no render is needed to exercise the
grading, retry and resume paths.

The resume tests (``ResumeFingerprintTest``) run a private *copy* of the
pipeline: ``batch_generate.sh`` fingerprints the sources that decide the spec,
so a parallel edit of the working tree could otherwise flip a "resume"
expectation into a re-run while the test runs.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
HOSTPY = sys.executable

# The pipeline seam is bash, so the fake is a small bash script around this
# python program.  Its behaviour comes from <outdir>/fake_mode:
#   ok            -> all checks pass (canonical_resolution follows PANOWORLD_RES)
#                    and the artifact files of a real export are written
#   depth_fail    -> depth_matches_geometry fails
#   files_missing -> files_present and every export check fail, no artifacts
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
"@PYTHON@" - "${OUT}" "${CANONICAL}" "${DEPTH}" "${FILES}" <<'PYEOF'
import json, os, sys

out, canonical, depth, files = sys.argv[1:5]
flags = {"canonical": canonical == "true", "files": files == "true", "depth": depth == "true"}
checks = [
    {"name": "canonical_resolution", "pass": flags["canonical"]},
    {"name": "files_present", "pass": flags["files"],
     "missing": [] if flags["files"] else ["scene/viewpoints"]},
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

# the files a real export_scene.py leaves behind: a panorama at the render
# resolution, a uint16 depth map with its scale, the camera pose, the viewpoint
# map and a topdown -- the batch resume gate decodes and measures all of them
if flags["files"]:
    import numpy as np
    from PIL import Image

    with open(os.path.join(out, "spec.json"), encoding="utf-8") as handle:
        spec = json.load(handle)
    width, height = (int(part) for part in os.environ.get("PANOWORLD_RES", "1024x512").split("x"))
    scene_dir = os.path.join(out, "scene")
    viewpoints = os.path.join(scene_dir, "viewpoints")
    os.makedirs(viewpoints, exist_ok=True)
    rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    names = []
    for camera in spec["cameras"]:
        name = camera["name"]
        names.append(name)
        view = os.path.join(viewpoints, name)
        os.makedirs(view, exist_ok=True)
        Image.new("RGB", (width, height), (120, 120, 120)).save(
            os.path.join(view, "panoImage_2048.png"))
        Image.fromarray(np.full((height, width), 1000, dtype=np.uint16)).save(
            os.path.join(view, "place_depth.png"))
        with open(os.path.join(view, "place_depth_scale.txt"), "w", encoding="utf-8") as handle:
            handle.write("1000\\n")
        x, y, z = (float(value) for value in camera["pos"])
        matrix = np.array([list(rotation[0]) + [x], list(rotation[1]) + [y],
                           list(rotation[2]) + [z], [0.0, 0.0, 0.0, 1.0]])
        np.savetxt(os.path.join(view, "extrinsics.txt"), matrix, fmt="%.10f")
    with open(os.path.join(scene_dir, "map.json"), "w", encoding="utf-8") as handle:
        json.dump({name: [] for name in names}, handle)
    Image.new("RGB", (64, 64), (200, 200, 200)).save(os.path.join(scene_dir, "topdown.png"))
PYEOF
exit 0
"""

FROZEN_ROOT = None


def frozen_repo_root():
    """A copy of the pipeline the tests can edit freely.

    ``batch_generate.sh`` binds a bundle to the fingerprints of the sources
    that produced it, so a resume test has to run against sources nobody else
    edits: a parallel agent saving ``layout_gen.py`` mid-test would correctly
    turn an expected "resume" into a re-run."""
    global FROZEN_ROOT
    if FROZEN_ROOT is None:
        root = tempfile.mkdtemp(prefix="batch_stats_frozen_")
        for name in ("blender_agent", "configs", "scripts"):
            shutil.copytree(os.path.join(REPO_ROOT, name), os.path.join(root, name),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        os.makedirs(os.path.join(root, "tools"))
        shutil.copy2(os.path.join(REPO_ROOT, "tools", "validate_blender_agent_p0.py"),
                     os.path.join(root, "tools", "validate_blender_agent_p0.py"))
        index_dir = os.path.join(root, "third_party", "assets", "polyhaven")
        os.makedirs(index_dir)
        shutil.copy2(os.path.join(REPO_ROOT, "third_party", "assets", "polyhaven", "index.json"),
                     os.path.join(index_dir, "index.json"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        FROZEN_ROOT = root
    return FROZEN_ROOT


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
    """Drives the scripts of one pipeline checkout (``repo``)."""

    repo = REPO_ROOT

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="batch_stats_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        with open(os.path.join(self.tmp, "fake_pipeline.sh"), "w", encoding="utf-8") as handle:
            handle.write(FAKE_PIPELINE.replace("@PYTHON@", HOSTPY))
        self.fake_pipeline = os.path.join(self.tmp, "fake_pipeline.sh")

    def script(self, name):
        return os.path.join(self.repo, "scripts", name)

    def run_layout(self, seed, out, rooms=None, env=None):
        args = ["bash", self.script("layout_gen_run.sh"), str(seed), out]
        if rooms is not None:
            args.append(str(rooms))
        return subprocess.run(args, capture_output=True, text=True, cwd=self.repo,
                              env=env or base_env())

    def run_batch(self, start, count, root, env=None):
        return subprocess.run(["bash", self.script("batch_generate.sh"), str(start), str(count), root],
                              capture_output=True, text=True, cwd=self.repo,
                              env=env or base_env())

    def summary_of(self, out):
        return load_json(os.path.join(out, "run_summary.json"))

    def batch_summary_of(self, root):
        return load_json(os.path.join(root, "batch_summary.json"))

    def manifest_of(self, root, seed=0):
        return load_json(os.path.join(root, "scene_%04d" % seed, "resume_manifest.json"))


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

    def test_a_degraded_retry_records_the_requested_and_effective_profile(self):
        """A retry that only succeeds at half the resolution is a bundle, but it
        is never reported as a completion at the requested profile."""
        root = os.path.join(self.tmp, "degraded")
        os.makedirs(os.path.join(root, "scene_0000"))
        with open(os.path.join(root, "scene_0000", "fake_mode"), "w", encoding="utf-8") as handle:
            handle.write("oom_then_ok")
        env = base_env(PANOWORLD_BATCH_RENDER=1, PANOWORLD_RES="1024x512",
                       PANOWORLD_SAMPLES=24, PANOWORLD_MIN_FREE_MB=1,
                       PANOWORLD_PIPELINE_SH=self.fake_pipeline)
        result = self.run_batch(0, 1, root, env=env)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = self.batch_summary_of(root)
        row = summary["scenes"][0]
        self.assertTrue(row["ok"]["bundle_ok"])
        quality = row["render_quality"]
        self.assertEqual(quality["requested"], {"resolution": "1024x512", "samples": 24})
        self.assertEqual(quality["effective"]["resolution"], "512x256")
        self.assertEqual(quality["effective"]["samples"], 16)
        self.assertTrue(quality["degraded"])
        self.assertEqual(summary["degraded"], 1)
        self.assertEqual(summary["degraded_seeds"], [0])
        self.assertIn("degraded 512x256, 16 samples at the requested 1024x512, 24 samples",
                      result.stdout)
        manifest = self.manifest_of(root)
        self.assertTrue(manifest["render"]["degraded"])
        self.assertEqual(manifest["render"]["effective_profile"]["resolution"], "512x256")
        # the artifact contract follows the profile that really rendered
        self.assertEqual(manifest["artifacts"]["problems"], [])
        with Image.open(os.path.join(root, "scene_0000", "scene", "viewpoints",
                                     manifest["artifacts"]["viewpoints"][0],
                                     "panoImage_2048.png")) as pano:
            self.assertEqual(pano.size, (512, 256))

    def test_summary_only_flags_a_camera_directory_without_artifacts(self):
        """An empty camera directory must not be counted as a complete bundle."""
        root = os.path.join(self.tmp, "empty_camera")
        scene = os.path.join(root, "scene_0000")
        os.makedirs(os.path.join(scene, "scene", "viewpoints", "0000"))
        with open(os.path.join(scene, "spec.json"), "w", encoding="utf-8") as handle:
            json.dump({"rooms": [{"id": "r0", "type": "living_room"}],
                       "cameras": [{"name": "0000", "room": "r0", "pos": [1.0, 1.0, 1.5]}]}, handle)
        with open(os.path.join(scene, "checks_report.json"), "w", encoding="utf-8") as handle:
            json.dump({"pass": True, "checks": [], "violations": []}, handle)
        with open(os.path.join(scene, "validation_report.json"), "w", encoding="utf-8") as handle:
            json.dump({"pass": True, "checks": [{"name": "files_present", "pass": True}]}, handle)
        with open(os.path.join(scene, "run_summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"kind": "layout_gen_run", "schema_version": 2, "run_id": "old-run",
                       "render_requested": True,
                       "stages": {"layout": "ok", "checks": "ok", "render": "ok",
                                  "validation": "ok"},
                       "ok": {"layout_ok": True, "render_ok": True, "validation_ok": True,
                              "bundle_ok": True},
                       "render": {"status": "ok", "resolution": "1024x512", "samples": 8,
                                  "attempts": [{"attempt": 1, "resolution": "1024x512",
                                                "samples": 8, "status": "ok"}]},
                       "checks": {"pass": True}, "rooms": ["r0"], "cameras": ["0000"]}, handle)
        result = self.run_batch(0, 1, root, env=base_env(PANOWORLD_BATCH_RENDER=1,
                                                         PANOWORLD_BATCH_SUMMARY_ONLY=1))
        self.assertEqual(result.returncode, 1, result.stderr[-2000:])
        summary = self.batch_summary_of(root)
        row = summary["scenes"][0]
        self.assertFalse(row["ok"]["bundle_ok"])
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["failure"]["kind"], "bundle_artifacts_missing")
        self.assertIn("scene/viewpoints/0000 (empty)", row["failure"]["detail"])
        self.assertEqual(summary["passed"], 0)


class ResumeFingerprintTest(ScriptCase):
    """Resume is bound to the request, the code/assets and the artifacts.

    The counterexamples of the review: a summary recording 512x256/4 samples
    with an empty camera directory must not satisfy a 2048x1024/128 request,
    and neither a changed render profile nor a changed layout request (floors,
    shape, rooms), a changed spec, changed sources or a missing artifact file
    may be reused."""

    def setUp(self):
        super().setUp()
        self.repo = frozen_repo_root()

    def render_env(self, **overrides):
        env = base_env(PANOWORLD_BATCH_RENDER=1, PANOWORLD_RES="1024x512",
                       PANOWORLD_SAMPLES=8, PANOWORLD_MIN_FREE_MB=1,
                       PANOWORLD_PIPELINE_SH=self.fake_pipeline)
        for key, value in overrides.items():
            env[key] = str(value)
        return env

    def resume_env(self, env):
        return dict(env, PANOWORLD_BATCH_RESUME="1")

    def scene(self, root, seed=0):
        return os.path.join(root, "scene_%04d" % seed)

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

    def test_a_matching_request_is_resumed_and_the_manifest_records_it(self):
        root = os.path.join(self.tmp, "match")
        env = self.render_env(PANOWORLD_ROOMS=4)
        first = self.run_batch(0, 1, root, env=env)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        manifest = self.manifest_of(root)
        self.assertEqual(manifest["kind"], "batch_scene_manifest")
        self.assertEqual(manifest["seed"], 0)
        self.assertEqual(manifest["request"]["layout"],
                         {"floors": "1", "shape": "row", "rooms": 4})
        self.assertEqual(manifest["request"]["render"],
                         {"requested": True, "resolution": "1024x512", "samples": 8})
        self.assertEqual(manifest["request"]["review"]["requested"], False)
        self.assertEqual(len(manifest["request"]["code"]["sha256"]), 64)
        self.assertIn("blender_agent/layout_gen.py", manifest["request"]["code"]["files"])
        self.assertTrue(manifest["request"]["assets"]["sha256"])
        self.assertEqual(len(manifest["request_sha256"]), 64)
        spec = load_json(os.path.join(self.scene(root), "spec.json"))
        self.assertEqual(manifest["spec"]["cameras"], [camera["name"] for camera in spec["cameras"]])
        self.assertEqual(manifest["spec"]["rooms"], len(spec["rooms"]))
        self.assertEqual(manifest["spec"]["floors"], 1)
        self.assertEqual(manifest["spec"]["shape"], "row")
        self.assertEqual(manifest["render"]["effective_profile"]["resolution"], "1024x512")
        self.assertFalse(manifest["render"]["degraded"])
        self.assertEqual(manifest["artifacts"]["problems"], [])
        self.assertTrue(manifest["bundle_ok"])

        second = self.run_batch(0, 1, root, env=self.resume_env(env))
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        row = self.batch_summary_of(root)["scenes"][0]
        self.assertTrue(row["resumed"], second.stdout[-2000:])
        self.assertIn("resumed: run", second.stdout)
        self.assertEqual(row["run_id"], manifest["run_id"])

    def test_a_changed_render_profile_is_rerun(self):
        root = os.path.join(self.tmp, "profile_change")
        env = self.render_env(PANOWORLD_ROOMS=4)
        first = self.run_batch(0, 1, root, env=env)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        old_run_id = self.batch_summary_of(root)["scenes"][0]["run_id"]
        old_pano = os.path.join(self.scene(root), "scene", "viewpoints")
        with Image.open(os.path.join(old_pano, os.listdir(old_pano)[0], "panoImage_2048.png")) as pano:
            self.assertEqual(pano.size, (1024, 512))

        new_env = self.render_env(PANOWORLD_ROOMS=4, PANOWORLD_RES="2048x1024",
                                  PANOWORLD_SAMPLES=128)
        second = self.run_batch(0, 1, root, env=self.resume_env(new_env))
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        row = self.batch_summary_of(root)["scenes"][0]
        self.assertFalse(row["resumed"], second.stdout[-2000:])
        self.assertNotEqual(row["run_id"], old_run_id)
        self.assertIn("request changed", second.stdout)
        self.assertIn("render.samples", second.stdout)
        manifest = self.manifest_of(root)
        self.assertEqual(manifest["request"]["render"]["resolution"], "2048x1024")
        self.assertEqual(manifest["request"]["render"]["samples"], 128)
        self.assertEqual(manifest["render"]["requested_profile"],
                         {"resolution": "2048x1024", "samples": 128})
        # the artifacts were really re-rendered at the requested profile
        viewpoints = os.path.join(self.scene(root), "scene", "viewpoints")
        for name in os.listdir(viewpoints):
            with Image.open(os.path.join(viewpoints, name, "panoImage_2048.png")) as pano:
                self.assertEqual(pano.size, (2048, 1024))
        self.assertEqual(manifest["artifacts"]["problems"], [])

    def test_a_changed_layout_request_is_rerun(self):
        root = os.path.join(self.tmp, "layout_change")
        env = base_env(PANOWORLD_LAYOUT_FLOORS=1, PANOWORLD_LAYOUT_SHAPE="row")
        first = self.run_batch(0, 1, root, env=env)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        self.assertEqual(self.manifest_of(root)["spec"]["floors"], 1)
        self.assertEqual(self.manifest_of(root)["request"]["layout"]["shape"], "row")

        # 1 -> 2 floors: the same seed, the same everything else
        duplex = self.run_batch(0, 1, root, env=dict(
            base_env(PANOWORLD_LAYOUT_FLOORS=2, PANOWORLD_LAYOUT_SHAPE="row"),
            PANOWORLD_BATCH_RESUME="1"))
        self.assertEqual(duplex.returncode, 0, duplex.stderr[-2000:])
        row = self.batch_summary_of(root)["scenes"][0]
        self.assertFalse(row["resumed"], duplex.stdout[-2000:])
        self.assertIn("layout.floors", duplex.stdout)
        spec = load_json(os.path.join(self.scene(root), "spec.json"))
        self.assertEqual(len(spec["floors"]), 2)
        manifest = self.manifest_of(root)
        self.assertEqual(manifest["request"]["layout"],
                         {"floors": "2", "shape": "row", "rooms": None})
        self.assertEqual(manifest["spec"]["floors"], 2)
        self.assertEqual(manifest["spec"]["shape"], "duplex")

        # back to one floor, and to another topology family
        again = self.run_batch(0, 1, root, env=dict(
            base_env(PANOWORLD_LAYOUT_FLOORS=1, PANOWORLD_LAYOUT_SHAPE="block"),
            PANOWORLD_BATCH_RESUME="1"))
        self.assertEqual(again.returncode, 0, again.stderr[-2000:])
        self.assertIn("layout.floors", again.stdout)
        self.assertIn("layout.shape", again.stdout)
        row = self.batch_summary_of(root)["scenes"][0]
        self.assertFalse(row["resumed"], again.stdout[-2000:])
        manifest = self.manifest_of(root)
        self.assertEqual(manifest["request"]["layout"],
                         {"floors": "1", "shape": "block", "rooms": None})
        self.assertEqual(manifest["spec"]["floors"], 1)
        self.assertEqual(manifest["spec"]["shape"], "block")

    def test_a_bundle_without_a_manifest_is_never_reused(self):
        """The review's counterexample: an old 512x256 / 4 sample success with
        an empty camera directory, resumed as 2048x1024 / 128 samples."""
        root = os.path.join(self.tmp, "old_summary")
        scene = self.scene(root)
        os.makedirs(os.path.join(scene, "scene", "viewpoints", "0000"))
        with open(os.path.join(scene, "spec.json"), "w", encoding="utf-8") as handle:
            json.dump({"rooms": [{"id": "r0", "type": "living_room"}],
                       "cameras": [{"name": "0000", "room": "r0", "pos": [1.0, 1.0, 1.5]}]}, handle)
        with open(os.path.join(scene, "checks_report.json"), "w", encoding="utf-8") as handle:
            json.dump({"pass": True, "checks": [], "violations": []}, handle)
        with open(os.path.join(scene, "validation_report.json"), "w", encoding="utf-8") as handle:
            json.dump({"pass": True, "checks": [{"name": "files_present", "pass": True}]}, handle)
        with open(os.path.join(scene, "run_summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"kind": "layout_gen_run", "schema_version": 2, "run_id": "old-512x256",
                       "render_requested": True,
                       "stages": {"layout": "ok", "checks": "ok", "render": "ok",
                                  "validation": "ok"},
                       "ok": {"layout_ok": True, "render_ok": True, "validation_ok": True,
                              "bundle_ok": True},
                       "render": {"status": "ok", "resolution": "512x256", "samples": 4,
                                  "attempts": [{"attempt": 1, "resolution": "512x256",
                                                "samples": 4, "status": "ok"}]},
                       "checks": {"pass": True}, "rooms": ["r0"], "cameras": ["0000"]}, handle)
        env = self.resume_env(self.render_env(PANOWORLD_ROOMS=4, PANOWORLD_RES="2048x1024",
                                              PANOWORLD_SAMPLES=128))
        result = self.run_batch(0, 1, root, env=env)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("no resume_manifest.json", result.stdout)
        row = self.batch_summary_of(root)["scenes"][0]
        self.assertFalse(row["resumed"], result.stdout[-2000:])
        self.assertNotEqual(row["run_id"], "old-512x256")
        self.assertEqual(row["run_id"], "%s-seed0" % self.batch_summary_of(root)["batch_id"])
        # the scene was really re-generated and rendered at the new request
        manifest = self.manifest_of(root)
        self.assertEqual(manifest["request"]["render"]["resolution"], "2048x1024")
        self.assertEqual(manifest["render"]["requested_profile"],
                         {"resolution": "2048x1024", "samples": 128})
        self.assertEqual(manifest["render"]["effective_profile"]["resolution"], "2048x1024")
        self.assertTrue(row["ok"]["bundle_ok"])

    def test_an_emptied_camera_directory_is_repaired_before_reuse(self):
        root = os.path.join(self.tmp, "empty_camera")
        env = self.render_env(PANOWORLD_ROOMS=4)
        first = self.run_batch(0, 1, root, env=env)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        scene = self.scene(root)
        cameras = [camera["name"] for camera in load_json(os.path.join(scene, "spec.json"))["cameras"]]
        self.assertGreater(len(cameras), 1)
        victim = cameras[1]
        view = os.path.join(scene, "scene", "viewpoints", victim)
        for name in os.listdir(view):
            os.remove(os.path.join(view, name))

        second = self.run_batch(0, 1, root, env=self.resume_env(env))
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        self.assertIn("artifacts incomplete", second.stdout)
        self.assertIn("scene/viewpoints/%s" % victim, second.stdout)
        row = self.batch_summary_of(root)["scenes"][0]
        self.assertFalse(row["resumed"], second.stdout[-2000:])
        self.assertTrue(row["ok"]["bundle_ok"], "the re-run rebuilt the artifacts")
        self.assertTrue(os.listdir(view))

        third = self.run_batch(0, 1, root, env=self.resume_env(env))
        self.assertEqual(third.returncode, 0, third.stderr[-2000:])
        self.assertTrue(self.batch_summary_of(root)["scenes"][0]["resumed"],
                        third.stdout[-2000:])

    def test_a_changed_spec_is_rerun(self):
        root = os.path.join(self.tmp, "spec_change")
        env = base_env(PANOWORLD_ROOMS=4)
        first = self.run_batch(0, 1, root, env=env)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        before = self.manifest_of(root)["spec"]["sha256"]
        spec_path = os.path.join(self.scene(root), "spec.json")
        spec = load_json(spec_path)
        spec["wall_height"] = 2.9
        with open(spec_path, "w", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=2)

        second = self.run_batch(0, 1, root, env=dict(env, PANOWORLD_BATCH_RESUME="1"))
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        row = self.batch_summary_of(root)["scenes"][0]
        self.assertFalse(row["resumed"], second.stdout[-2000:])
        self.assertIn("spec.json changed", second.stdout)
        self.assertEqual(load_json(spec_path)["wall_height"], 2.7)
        self.assertEqual(self.manifest_of(root)["spec"]["sha256"], before,
                         "the re-generated spec must hash to the recorded value again")

    def test_a_panorama_of_the_wrong_size_is_rerun(self):
        root = os.path.join(self.tmp, "wrong_size")
        env = self.render_env(PANOWORLD_ROOMS=4)
        first = self.run_batch(0, 1, root, env=env)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        scene = self.scene(root)
        camera = load_json(os.path.join(scene, "spec.json"))["cameras"][0]["name"]
        # a decodable panorama that is not the render resolution any more
        Image.new("RGB", (64, 32), (10, 10, 10)).save(
            os.path.join(scene, "scene", "viewpoints", camera, "panoImage_2048.png"))

        second = self.run_batch(0, 1, root, env=self.resume_env(env))
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        self.assertIn("artifacts incomplete", second.stdout)
        self.assertIn("expected", second.stdout)
        row = self.batch_summary_of(root)["scenes"][0]
        self.assertFalse(row["resumed"], second.stdout[-2000:])
        with Image.open(os.path.join(scene, "scene", "viewpoints", camera,
                                     "panoImage_2048.png")) as pano:
            self.assertEqual(pano.size, (1024, 512))

    def test_a_changed_asset_registry_is_rerun(self):
        root = os.path.join(self.tmp, "asset_change")
        env = base_env(PANOWORLD_ROOMS=4)
        first = self.run_batch(0, 1, root, env=env)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        index = os.path.join(self.repo, "third_party", "assets", "polyhaven", "index.json")
        with open(index, "rb") as handle:
            original = handle.read()
        self.addCleanup(self._restore, index, original)
        with open(index, "ab") as handle:
            handle.write(b"\n")

        second = self.run_batch(0, 1, root, env=dict(env, PANOWORLD_BATCH_RESUME="1"))
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        row = self.batch_summary_of(root)["scenes"][0]
        self.assertFalse(row["resumed"], second.stdout[-2000:])
        self.assertIn("assets.sha256", second.stdout)

    def test_z_code_change_is_rerun(self):
        root = os.path.join(self.tmp, "code_change")
        env = base_env(PANOWORLD_ROOMS=4)
        first = self.run_batch(0, 1, root, env=env)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        resumed = self.run_batch(0, 1, root, env=dict(env, PANOWORLD_BATCH_RESUME="1"))
        self.assertEqual(resumed.returncode, 0, resumed.stderr[-2000:])
        self.assertTrue(self.batch_summary_of(root)["scenes"][0]["resumed"],
                        resumed.stdout[-2000:])

        # the same edit a parallel agent would make to the generator
        source = os.path.join(self.repo, "blender_agent", "layout_gen.py")
        with open(source, "rb") as handle:
            original = handle.read()
        self.addCleanup(self._restore, source, original)
        with open(source, "ab") as handle:
            handle.write(b"\n# fingerprint probe\n")
        changed = self.run_batch(0, 1, root, env=dict(env, PANOWORLD_BATCH_RESUME="1"))
        self.assertEqual(changed.returncode, 0, changed.stderr[-2000:])
        row = self.batch_summary_of(root)["scenes"][0]
        self.assertFalse(row["resumed"], changed.stdout[-2000:])
        self.assertIn("request changed", changed.stdout)
        self.assertIn("code.files.blender_agent/layout_gen.py", changed.stdout)

    @staticmethod
    def _restore(path, content):
        with open(path, "wb") as handle:
            handle.write(content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
