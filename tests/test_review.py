"""Visual/rule review and the review -> repair -> re-check loop.

Run::

    python tests/test_review.py            # or: python tests/test_review.py -v

Three layers are covered:

* `blender_agent/review.py`'s checks, on synthetic packages built here (a spec
  with an empty kitchen, a manifest with furniture that settled floating, a
  panorama that is one white blob, a build made from another spec ...);
* the repair path, which must turn the review's hints into spec tweaks that
  still pass ``checks.py``;
* the shell hooks: ``scripts/layout_gen_run.sh`` runs the review before and
  after the render, grades ``review_ok``/exit code 6 and repairs once when
  asked, ``scripts/batch_generate.sh`` aggregates the extra level.  The Blender
  pipeline is replaced by a fake through ``PANOWORLD_PIPELINE_SH`` that writes
  render artifacts (and, in one test, damages the spec so the closed loop has
  something to repair), so no render is needed.
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
sys.path.insert(0, REPO_ROOT)

from blender_agent import checks as checks_mod  # noqa: E402
from blender_agent import layout_gen as layout_mod  # noqa: E402
from blender_agent import review as review_mod  # noqa: E402

HOSTPY = sys.executable
LAYOUT_RUN = os.path.join(REPO_ROOT, "scripts", "layout_gen_run.sh")
BATCH = os.path.join(REPO_ROOT, "scripts", "batch_generate.sh")

try:                                            # the pixel checks need numpy + PIL
    import numpy  # noqa: F401
    from PIL import Image

    HAVE_IMAGES = True
except Exception:                               # pragma: no cover - host without PIL
    HAVE_IMAGES = False

#: Fake Blender pipeline: the Python program it wraps gets the spec and the
#: outdir and writes a validation report plus -- per mode -- build artifacts.
#:   ok            all checks pass, grey panoramas, settled manifest
#:   floating      one floor-resting piece settles 8 cm up
#:   blown         every panorama is pure white
#:   dark          every panorama is nearly black
#:   damage_spec   the first call strips the furniture of the first room, so
#:                 the post-render review fails and the repair loop must fix it
#:   files_missing no viewpoints at all (render graded as failed)
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
"@PYTHON@" - "${SPEC}" "${OUT}" "${MODE}" "${CALLS}" <<'PYEOF'
import json, os, sys

spec_path, out, mode, calls = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
with open(spec_path, encoding="utf-8") as handle:
    spec = json.load(handle)

if mode == "damage_spec" and calls == 1:
    rooms = [room["id"] for room in spec["rooms"]]
    target = rooms[0]
    spec["furniture"] = [item for item in spec["furniture"] if item["room"] != target]
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)

files_ok = mode != "files_missing"
checks = [
    {"name": "canonical_resolution", "pass": False},
    {"name": "files_present", "pass": files_ok, "missing": [] if files_ok else ["scene/viewpoints"]},
    {"name": "depth_matches_geometry", "pass": True, "cameras": []},
    {"name": "door_openings_geometry", "pass": True, "openings": len(spec.get("openings", []))},
]
with open(os.path.join(out, "validation_report.json"), "w", encoding="utf-8") as handle:
    json.dump({"kind": "blender_agent_p0_validation",
               "pass": all(check["pass"] for check in checks), "checks": checks}, handle)

if files_ok:
    try:
        from PIL import Image
    except Exception:
        Image = None
    for camera in spec["cameras"]:
        view = os.path.join(out, "scene", "viewpoints", camera["name"])
        os.makedirs(view, exist_ok=True)
        if Image is not None:
            colour = {"blown": (255, 255, 255), "dark": (8, 8, 8)}.get(mode, (120, 120, 120))
            Image.new("RGB", (64, 32), colour).save(os.path.join(view, "panoImage_2048.png"))
        with open(os.path.join(view, "place_depth_scale.txt"), "w") as handle:
            handle.write("1000\\n")

    floating_done = False
    items = []
    for index, item in enumerate(spec.get("furniture", [])):
        settle = {"mode": "rest", "grounded": True, "penetrating": False, "flat": False,
                  "bottom_gap_m": 0.0}
        if item.get("asset") == "builtin:rug":
            settle["flat"] = True
        if item.get("mount") == "ceiling":
            settle["mode"] = "held"
            settle["bottom_gap_m"] = None
        if mode == "floating" and settle["mode"] == "rest" and not settle["flat"] and not floating_done:
            settle["grounded"] = False
            settle["bottom_gap_m"] = 0.08
            floating_done = True
        items.append({"name": item.get("name") or f"item_{index}", "asset": item.get("asset"),
                      "kind": "builtin" if str(item.get("asset", "")).startswith("builtin:") else "asset",
                      "mount": item.get("mount", "floor"), "pos": item["pos"],
                      "bbox_z": [0.0, 0.5], "settle": settle})
    manifest = {"kind": "blender_agent_build", "spec_path": spec_path, "spec": spec,
                "render": {"width": 1024, "height": 512, "samples": 24},
                "furniture": items, "cameras": [{"camera": camera["name"], "pos": camera["pos"] + [1.5]}
                                                for camera in spec["cameras"]]}
    os.makedirs(os.path.join(out, "build"), exist_ok=True)
    with open(os.path.join(out, "build", "build_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
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


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def generated_spec(seed: int = 5, shape: str = "compact_2x2", rooms: int = None) -> dict:
    """A real, valid spec from the generator (fast: no file round-trip)."""
    index = checks_mod.load_assets_index()
    scene = layout_mod.generate(seed, rooms, index, attempts=layout_mod.MAX_LAYOUT_ATTEMPTS,
                                verbose=False, shape=shape)
    return scene.spec


def spec_without_room_furniture(spec: dict, room_type: str) -> dict:
    """Drop every furniture item of the first room of that type (an empty room)."""
    payload = json.loads(json.dumps(spec))
    target = next(room["id"] for room in payload["rooms"] if room["type"] == room_type)
    payload["furniture"] = [item for item in payload["furniture"] if item["room"] != target]
    return payload


class ScenePackage:
    """A throwaway scene directory: spec.json (+ optional build artifacts)."""

    def __init__(self, spec: dict, manifest: dict = None, panorama=None, root: str = None) -> None:
        self.root = root or tempfile.mkdtemp(prefix="review_")
        write_json(os.path.join(self.root, "spec.json"), spec)
        if manifest is not None:
            os.makedirs(os.path.join(self.root, "build"), exist_ok=True)
            write_json(os.path.join(self.root, "build", "build_manifest.json"), manifest)
        if panorama is not None:
            for camera in spec["cameras"]:
                view = os.path.join(self.root, "scene", "viewpoints", camera["name"])
                os.makedirs(view, exist_ok=True)
                panorama(os.path.join(view, "panoImage_2048.png"))

    def check(self, name: str, report: dict = None) -> dict:
        report = report or review_mod.review(self.root)
        return next(check for check in report["checks"] if check["name"] == name)


def manifest_for(spec: dict, gap_m: float = 0.0, flat_asset: str = "builtin:rug") -> dict:
    items = []
    for index, item in enumerate(spec.get("furniture", [])):
        settle = {"mode": "rest", "grounded": True, "penetrating": False,
                  "flat": item.get("asset") == flat_asset, "bottom_gap_m": gap_m}
        if item.get("mount") == "ceiling":
            settle.update({"mode": "held", "bottom_gap_m": None})
        items.append({"name": item.get("name") or f"item_{index}", "asset": item.get("asset"),
                      "mount": item.get("mount", "floor"), "settle": settle})
    return {"kind": "blender_agent_build", "spec": spec, "furniture": items, "cameras": []}


def grey_panorama(path: str) -> None:
    if not HAVE_IMAGES:
        return
    Image.new("RGB", (64, 32), (120, 120, 120)).save(path)


def white_blob_panorama(path: str) -> None:
    if not HAVE_IMAGES:
        return
    image = Image.new("RGB", (64, 32), (120, 120, 120))
    for x in range(0, 32):
        for y in range(0, 32):
            image.putpixel((x, y), (255, 255, 255))
    image.save(path)


class CheckTest(unittest.TestCase):
    """One defect per check, on a real generated spec."""

    def setUp(self):
        self.spec = generated_spec()
        self.index = checks_mod.load_assets_index()

    def scene(self, spec: dict, **kwargs) -> ScenePackage:
        package = ScenePackage(spec, **kwargs)
        self.addCleanup(shutil.rmtree, package.root, ignore_errors=True)
        return package

    def test_clean_spec_passes_and_reports_every_check(self):
        package = self.scene(self.spec)
        report = review_mod.review(package.root, index=self.index)
        names = [check["name"] for check in report["checks"]]
        for expected in ("rooms_furnished", "functional_coverage", "no_duplicate_furniture",
                         "camera_clearance", "camera_framing", "camera_coverage"):
            self.assertIn(expected, names)
        self.assertTrue(report["pass"], report["issues"])
        self.assertEqual(report["stats"]["issues_fail"], 0)
        self.assertFalse(report["scopes"]["build"], "a spec-only package has no build scope")

    def test_empty_habitable_room_fails_with_a_repair_hint(self):
        spec = spec_without_room_furniture(self.spec, "kitchen")
        empty_room = next(room["id"] for room in spec["rooms"] if room["type"] == "kitchen")
        package = self.scene(spec)
        report = review_mod.review(package.root, index=self.index)
        self.assertFalse(report["pass"])
        kinds = {issue["kind"] for issue in report["issues"] if issue["severity"] == "fail"}
        self.assertIn("empty_room", kinds)
        self.assertIn("missing_kitchen", kinds)
        hints = [issue["repair"] for issue in report["issues"] if issue["kind"] == "empty_room"]
        self.assertEqual(hints, [{"kind": "add_furniture", "room": empty_room, "room_type": "kitchen"}])

    def test_functional_gap_is_reported_per_room_type(self):
        spec = json.loads(json.dumps(self.spec))
        bathroom = next(room["id"] for room in spec["rooms"] if room["type"] == "bathroom")
        spec["furniture"] = [item for item in spec["furniture"]
                             if not (item["room"] == bathroom
                                     and item["asset"] == "builtin:bath_toilet")]
        package = self.scene(spec)
        report = review_mod.review(package.root, index=self.index)
        detail = " ".join(issue["detail"] for issue in report["issues"]
                          if issue.get("room") == bathroom)
        self.assertIn("no toilet", detail)

    def test_duplicate_placement_fails_and_repeats_warn(self):
        spec = json.loads(json.dumps(self.spec))
        room = spec["rooms"][0]["id"]
        source = next(item for item in spec["furniture"] if item["room"] == room)
        twin = dict(source, name="twin")
        spec["furniture"].append(twin)
        package = self.scene(spec)
        report = review_mod.review(package.root, index=self.index)
        kinds = {issue["kind"] for issue in report["issues"]}
        self.assertIn("duplicate_placement", kinds)
        self.assertFalse(report["pass"])

    def test_camera_in_a_wall_fails_and_hugging_warns(self):
        spec = json.loads(json.dumps(self.spec))
        room = next(room for room in spec["rooms"] if len(room["polygon"]) == 4)
        xs = [point[0] for point in room["polygon"]]
        ys = [point[1] for point in room["polygon"]]
        camera = next(entry for entry in spec["cameras"] if entry["room"] == room["id"])
        camera["pos"] = [min(xs) + 0.02, (min(ys) + max(ys)) / 2.0, 1.5]
        package = self.scene(spec)
        report = review_mod.review(package.root, index=self.index)
        kinds = {issue["kind"] for issue in report["issues"]}
        self.assertIn("camera_in_wall", kinds)
        self.assertIn("move_camera", [issue["repair"]["kind"] for issue in report["issues"]])

    def test_stale_build_fails(self):
        spec = json.loads(json.dumps(self.spec))
        stale = json.loads(json.dumps(self.spec))
        stale["wall_height"] = 3.1
        package = self.scene(spec, manifest=manifest_for(stale))
        report = review_mod.review(package.root, index=self.index)
        self.assertFalse(report["pass"])
        self.assertIn("stale_build", {issue["kind"] for issue in report["issues"] if
                                      issue["severity"] == "fail"})

    def test_floating_furniture_fails_and_flat_finish_is_exempt(self):
        spec = json.loads(json.dumps(self.spec))
        moving = next(item for item in spec["furniture"]
                      if item["asset"] != "builtin:rug" and item.get("mount", "floor") == "floor")
        # carpet the room with rugs: they move with the floor and must not fail
        spec["furniture"].append({"asset": "builtin:rug", "room": moving["room"],
                                  "pos": list(moving["pos"]), "size": [1.5, 1.0], "name": "rug_probe"})
        manifest = manifest_for(spec)
        for entry in manifest["furniture"]:
            if entry["name"] == moving.get("name"):
                entry["settle"].update({"grounded": False, "bottom_gap_m": 0.08})
            if entry["asset"] == "builtin:rug":
                entry["settle"].update({"flat": True, "grounded": False, "bottom_gap_m": -0.02,
                                        "penetrating": True})
        package = self.scene(spec, manifest=manifest)
        report = review_mod.review(package.root, index=self.index)
        fails = [issue for issue in report["issues"] if issue["severity"] == "fail"]
        self.assertEqual([issue["kind"] for issue in fails], ["furniture_floating"])
        grounding = package.check("furniture_grounding", report)
        self.assertFalse(grounding["pass"])

    @unittest.skipUnless(HAVE_IMAGES, "numpy/PIL not available")
    def test_exposure_flags_a_blown_and_a_dark_panorama(self):
        package = self.scene(self.spec, panorama=white_blob_panorama)
        report = review_mod.review(package.root, index=self.index)
        self.assertTrue(report["scopes"]["render"])
        exposure = package.check("exposure", report)
        self.assertFalse(exposure["pass"])
        self.assertEqual(exposure["issues"][0]["severity"], "fail")

        dark = self.scene(self.spec, panorama=lambda path: Image.new("RGB", (64, 32), (8, 8, 8)).save(path))
        report = review_mod.review(dark.root, index=self.index)
        exposure = dark.check("exposure", report)
        self.assertIn("dark_view", {issue["kind"] for issue in exposure["issues"]})
        self.assertEqual({issue["severity"] for issue in exposure["issues"]}, {"warn"})
        self.assertTrue(report["pass"], "a dark viewpoint is a warning, not a failure")

    @unittest.skipUnless(HAVE_IMAGES, "numpy/PIL not available")
    def test_grey_panorama_passes_exposure(self):
        package = self.scene(self.spec, panorama=grey_panorama)
        report = review_mod.review(package.root, index=self.index)
        self.assertTrue(package.check("exposure", report)["pass"])

    def test_missing_spec_is_an_error_report(self):
        root = tempfile.mkdtemp(prefix="review_missing_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        report = review_mod.review(root, index=self.index)
        self.assertEqual(report.get("error"), "missing_spec")
        self.assertFalse(report["pass"])

    def test_external_reviewer_is_pluggable_and_cannot_crash_the_review(self):
        package = self.scene(self.spec, panorama=grey_panorama)
        verdict = os.path.join(package.root, "verdict.json")
        write_json(verdict, {"pass": True, "score": 0.5, "issues": [],
                             "suggestions": [{"kind": "note"}]})
        script = os.path.join(package.root, "reviewer.sh")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env bash\ncat '%s'\n" % verdict)
        os.chmod(script, 0o755)
        report = review_mod.review(package.root, index=self.index, review_cmd=f"bash {script}")
        self.assertEqual(report["external"]["status"], "ok")
        self.assertTrue(report["pass"])

        broken = os.path.join(package.root, "broken.sh")
        with open(broken, "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env bash\necho not json\nexit 3\n")
        os.chmod(broken, 0o755)
        report = review_mod.review(package.root, index=self.index, review_cmd=f"bash {broken}")
        self.assertEqual(report["external"]["status"], "error")
        self.assertTrue(report["pass"], "a broken hook must not fail the scene")

        rejecting = os.path.join(package.root, "reject.json")
        write_json(rejecting, {"pass": False, "issues": [{"detail": "looks wrong", "severity": "fail"}]})
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env bash\ncat '%s'\n" % rejecting)
        report = review_mod.review(package.root, index=self.index, review_cmd=f"bash {script}")
        self.assertFalse(report["pass"])
        self.assertEqual(report["issues"][-1]["source"], "external")

    def test_review_report_is_written_by_the_cli(self):
        package = self.scene(self.spec)
        result = subprocess.run([HOSTPY, os.path.join(REPO_ROOT, "blender_agent", "review.py"),
                                 package.root], capture_output=True, text=True, cwd=REPO_ROOT)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        report = load_json(os.path.join(package.root, "review_report.json"))
        self.assertEqual(report["kind"], "blender_agent_review")
        self.assertTrue(report["pass"])
        self.assertIn("review: PASS", result.stdout)


class RepairTest(unittest.TestCase):
    """The repair turns hints into spec tweaks that keep checks.py green."""

    def setUp(self):
        self.index = checks_mod.load_assets_index()

    def _scene(self, spec: dict) -> str:
        root = tempfile.mkdtemp(prefix="repair_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        write_json(os.path.join(root, "spec.json"), spec)
        return root

    def test_empty_kitchen_is_furnished_and_checks_still_pass(self):
        spec = spec_without_room_furniture(generated_spec(), "kitchen")
        root = self._scene(spec)
        report = review_mod.review(root, index=self.index)
        self.assertFalse(report["pass"])

        result = review_mod.apply_repair(root, report=report, index=self.index)
        self.assertTrue(result["changed"], result)
        self.assertTrue(result["checks"]["pass"], result["checks"])
        actions = {action["kind"] for action in result["actions"]}
        self.assertIn("add_furniture", actions)
        repaired = load_json(os.path.join(root, "spec.json"))
        self.assertEqual(len(repaired["furniture"]), len(spec["furniture"]) + 1)
        self.assertTrue(os.path.isfile(os.path.join(root, "spec.pre_review.json")))
        checks_report = checks_mod.check_file(os.path.join(root, "spec.json"))
        self.assertTrue(checks_report["pass"], checks_report["violations"])

    def test_repair_after_the_second_review_passes(self):
        spec = spec_without_room_furniture(generated_spec(seed=11), "bathroom")
        root = self._scene(spec)
        review_mod.apply_repair(root, report=review_mod.review(root, index=self.index), index=self.index)
        second = review_mod.review(root, index=self.index)
        kinds = {issue["kind"] for issue in second["issues"] if issue["severity"] == "fail"}
        self.assertNotIn("empty_room", kinds)
        self.assertNotIn("missing_bathroom", kinds)

    def test_dry_run_does_not_touch_the_spec(self):
        spec = spec_without_room_furniture(generated_spec(seed=3), "kitchen")
        root = self._scene(spec)
        report = review_mod.review(root, index=self.index)
        before = load_json(os.path.join(root, "spec.json"))
        result = review_mod.apply_repair(root, report=report, index=self.index, dry_run=True)
        self.assertTrue(result["changed"])
        self.assertEqual(load_json(os.path.join(root, "spec.json")), before)

    def test_nothing_to_repair_on_a_clean_scene(self):
        root = self._scene(generated_spec(seed=2))
        report = review_mod.review(root, index=self.index)
        result = review_mod.apply_repair(root, report=report, index=self.index)
        self.assertFalse(result["changed"])
        self.assertEqual(result["status"], "nothing_to_do")

    def test_apply_repair_reads_an_explicit_report_path(self):
        """The pre-render hook names its report (review_prerender.json)."""
        spec = spec_without_room_furniture(generated_spec(seed=7), "kitchen")
        root = self._scene(spec)
        pre = os.path.join(root, "review_prerender.json")
        write_json(pre, review_mod.review(root, index=self.index))
        self.assertFalse(os.path.isfile(os.path.join(root, "review_report.json")))
        result = review_mod.apply_repair(root, index=self.index, report_path=pre)
        self.assertTrue(result["changed"], result)
        repaired = load_json(os.path.join(root, "spec.json"))
        self.assertGreater(len(repaired["furniture"]), len(spec["furniture"]))


class HookTest(unittest.TestCase):
    """The shell hooks: review_ok grading, exit code 6, the one repair retry."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="review_hook_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        path = os.path.join(self.tmp, "fake_pipeline.sh")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(FAKE_PIPELINE.replace("@PYTHON@", HOSTPY))
        os.chmod(path, 0o755)
        self.fake_pipeline = path

    def run_layout(self, seed, out, env):
        return subprocess.run(["bash", LAYOUT_RUN, str(seed), out], capture_output=True,
                              text=True, cwd=REPO_ROOT, env=env)

    def run_batch(self, start, count, root, env):
        return subprocess.run(["bash", BATCH, str(start), str(count), root], capture_output=True,
                              text=True, cwd=REPO_ROOT, env=env)

    def mode_env(self, out, mode, **overrides):
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "fake_mode"), "w", encoding="utf-8") as handle:
            handle.write(mode)
        return base_env(PANOWORLD_RES="1024x512", PANOWORLD_SAMPLES=8,
                        PANOWORLD_MIN_FREE_MB=1, PANOWORLD_PIPELINE_SH=self.fake_pipeline,
                        PANOWORLD_REVIEW=1, **overrides)

    def summary_of(self, out):
        return load_json(os.path.join(out, "run_summary.json"))

    def test_layout_only_review_grades_review_ok(self):
        out = os.path.join(self.tmp, "layout_only")
        result = self.run_layout(0, out, base_env(PANOWORLD_REVIEW=1, PANOWORLD_SKIP_RENDER=1,
                                                 PANOWORLD_RUN_ID="unit-review-layout"))
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertIs(summary["ok"]["review_ok"], True)
        self.assertEqual(summary["stages"]["review"], "ok")
        self.assertEqual(summary["stages"]["render"], "skipped_by_request")
        self.assertTrue(summary["review"]["scopes"]["spec"])
        self.assertFalse(summary["review"]["scopes"]["render"])
        self.assertIsNone(summary["failure"])
        self.assertTrue(os.path.isfile(os.path.join(out, "review_report.json")))
        self.assertTrue(os.path.isfile(os.path.join(out, "review_prerender.json")))

    def test_good_render_passes_the_review_and_bundles(self):
        out = os.path.join(self.tmp, "good")
        env = self.mode_env(out, "ok", PANOWORLD_RUN_ID="unit-review-good")
        result = self.run_layout(0, out, env)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertEqual(summary["ok"], {"layout_ok": True, "render_ok": True, "validation_ok": True,
                                         "review_ok": True, "bundle_ok": True})
        self.assertEqual(summary["stages"]["review"], "ok")
        self.assertEqual(summary["review"]["issues"]["fail"], 0)

    def test_floating_furniture_fails_the_review_with_exit_6(self):
        out = os.path.join(self.tmp, "floating")
        env = self.mode_env(out, "floating", PANOWORLD_RUN_ID="unit-review-floating")
        result = self.run_layout(0, out, env)
        self.assertEqual(result.returncode, 6, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertIs(summary["ok"]["review_ok"], False)
        self.assertEqual(summary["stages"]["review"], "failed")
        self.assertFalse(summary["ok"]["bundle_ok"])
        self.assertEqual(summary["failure"]["stage"], "review")
        self.assertEqual(summary["failure"]["kind"], "review_failed")
        self.assertIn("furniture_grounding", summary["failure"]["checks"])
        self.assertEqual(summary["render"]["status"], "ok", "the render itself was complete")

    def test_blown_panorama_fails_the_review(self):
        out = os.path.join(self.tmp, "blown")
        env = self.mode_env(out, "blown", PANOWORLD_RUN_ID="unit-review-blown")
        result = self.run_layout(0, out, env)
        self.assertEqual(result.returncode, 6, result.stderr[-2000:])
        self.assertEqual(self.summary_of(out)["stages"]["review"], "failed")

    def test_review_can_be_disabled_and_leaves_the_old_summary_shape(self):
        out = os.path.join(self.tmp, "no_review")
        env = base_env(PANOWORLD_RES="1024x512", PANOWORLD_SAMPLES=8, PANOWORLD_MIN_FREE_MB=1,
                       PANOWORLD_PIPELINE_SH=self.fake_pipeline, PANOWORLD_RUN_ID="unit-no-review")
        result = self.run_layout(0, out, env)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = self.summary_of(out)
        self.assertEqual(summary["ok"], {"layout_ok": True, "render_ok": True, "validation_ok": True,
                                         "bundle_ok": True})
        self.assertNotIn("review", summary["stages"])
        self.assertFalse(os.path.isfile(os.path.join(out, "review_report.json")))

    def test_repair_retry_fixes_a_damaged_spec_and_re_renders(self):
        out = os.path.join(self.tmp, "repair_loop")
        env = self.mode_env(out, "damage_spec", PANOWORLD_REVIEW_REPAIR=1,
                            PANOWORLD_RUN_ID="unit-review-repair")
        result = self.run_layout(0, out, env)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        summary = self.summary_of(out)
        self.assertTrue(summary["ok"]["review_ok"])
        self.assertTrue(summary["ok"]["bundle_ok"])
        self.assertTrue(summary["review"]["repair"]["used"])
        self.assertEqual(summary["review"]["repair"]["status"], "applied")
        repair = load_json(os.path.join(out, "review_repair.json"))
        self.assertTrue(repair["changed"])
        self.assertIn("add_furniture", {action["kind"] for action in repair["actions"]})
        # the repaired spec was rendered again (3 attempts: 1 render, 1 repair re-render)
        statuses = [(attempt["attempt"], attempt["status"]) for attempt in summary["render"]["attempts"]]
        self.assertIn((3, "ok"), statuses)
        self.assertTrue(os.path.isfile(os.path.join(out, "spec.pre_review.json")))
        # the manifest the review finally judged was built from the repaired spec
        manifest = load_json(os.path.join(out, "build", "build_manifest.json"))
        self.assertEqual(manifest["spec"], load_json(os.path.join(out, "spec.json")))

    def test_batch_aggregates_the_review_level(self):
        root = os.path.join(self.tmp, "batch")
        os.makedirs(os.path.join(root, "scene_0001"), exist_ok=True)
        with open(os.path.join(root, "scene_0001", "fake_mode"), "w", encoding="utf-8") as handle:
            handle.write("floating")
        env = base_env(PANOWORLD_BATCH_RENDER=1, PANOWORLD_REVIEW=1, PANOWORLD_RES="1024x512",
                       PANOWORLD_SAMPLES=8, PANOWORLD_MIN_FREE_MB=1,
                       PANOWORLD_PIPELINE_SH=self.fake_pipeline)
        result = self.run_batch(0, 2, root, env)
        self.assertEqual(result.returncode, 1, result.stderr[-2000:])
        summary = load_json(os.path.join(root, "batch_summary.json"))
        self.assertTrue(summary["review_requested"])
        self.assertEqual(summary["levels"]["review_ok"], 1)
        self.assertEqual(summary["levels"]["bundle_ok"], 1)
        self.assertEqual(summary["review_issues"], {"furniture_grounding": 1})
        statuses = {row["seed"]: row["status"] for row in summary["scenes"]}
        self.assertEqual(statuses, {0: "bundle_ok", 1: "failed"})
        self.assertTrue(summary["scenes"][0]["review_pass"])
        self.assertFalse(summary["scenes"][1]["review_pass"])
        self.assertEqual(summary["scenes"][1]["failure"]["stage"], "review")

    def test_batch_without_review_keeps_the_four_level_shape(self):
        root = os.path.join(self.tmp, "batch_plain")
        result = self.run_batch(0, 2, root, base_env())
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = load_json(os.path.join(root, "batch_summary.json"))
        self.assertEqual(set(summary["levels"]),
                         {"layout_ok", "render_ok", "validation_ok", "bundle_ok"})
        self.assertFalse(summary["review_requested"])
        self.assertEqual(summary["review_issues"], {})
        for row in summary["scenes"]:
            self.assertNotIn("review_pass", row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
