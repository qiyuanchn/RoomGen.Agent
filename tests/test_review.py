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


_SPEC_CACHE: dict = {}


def cached_spec(seed: int = 5, shape: str = "compact_2x2") -> dict:
    """One generated spec per (seed, shape), deep-copied for each caller.

    Layout generation costs a second or so, and the exposure tests only need a
    real spec to hang their synthetic build on.
    """
    key = (seed, shape)
    if key not in _SPEC_CACHE:
        _SPEC_CACHE[key] = generated_spec(seed, shape)
    return json.loads(json.dumps(_SPEC_CACHE[key]))


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


def read_png(path: str):
    """A panorama as float RGB in 0..1, the way the review measures it."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype="float32") / 255.0


def pixel_luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def write_exr(path: str, rgb) -> None:
    """A minimal ZIP-compressed OpenEXR, the counterpart of ``review.read_exr``."""
    import struct
    import zlib

    height, width, _ = rgb.shape
    channels = b""
    for name in ("B", "G", "R"):
        channels += (name.encode() + b"\x00" + struct.pack("<i", 2) + b"\x00" + b"\x00" * 3
                     + struct.pack("<ii", 1, 1))
    channels += b"\x00"

    def attribute(name: str, kind: str, payload: bytes) -> bytes:
        return (name.encode() + b"\x00" + kind.encode() + b"\x00"
                + struct.pack("<i", len(payload)) + payload)

    header = bytearray(b"\x76\x2f\x31\x01" + struct.pack("<i", 2))
    header += attribute("channels", "chlist", channels)
    header += attribute("compression", "compression", b"\x03")          # ZIP
    header += attribute("dataWindow", "box2i", struct.pack("<4i", 0, 0, width - 1, height - 1))
    header += attribute("displayWindow", "box2i", struct.pack("<4i", 0, 0, width - 1, height - 1))
    header += attribute("lineOrder", "lineOrder", b"\x00")
    header += attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += attribute("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\x00"

    blocks = []
    for start in range(0, height, 16):
        raw = bytearray()
        for row in rgb[start:start + 16]:
            for channel in (2, 1, 0):                                   # chlist order B, G, R
                raw += row[:, channel].astype("<f4").tobytes()
        half = (len(raw) + 1) // 2
        interleaved = bytearray(len(raw))
        interleaved[:half] = raw[0::2]                                  # even bytes first
        interleaved[half:] = raw[1::2]
        predicted = bytearray(interleaved)
        for index in range(len(predicted) - 1, 0, -1):                  # predict from the end
            predicted[index] = (predicted[index] - predicted[index - 1] + 128) & 0xFF
        blocks.append(zlib.compress(bytes(predicted)))

    offset = len(header) + 8 * len(blocks)
    table = bytearray()
    body = bytearray()
    for index, block in enumerate(blocks):
        table += struct.pack("<Q", offset)
        chunk = struct.pack("<ii", index * 16, len(block)) + block
        body += chunk
        offset += len(chunk)
    with open(path, "wb") as handle:
        handle.write(bytes(header) + bytes(table) + bytes(body))


def synthetic_frames(height: int = 32, width: int = 64, blown_rows=None, blown: float = 1.6,
                     base: float = 0.25, dome_rows: int = 0, dome: float = 1.6,
                     room_depth: float = 2.5, dome_depth: float = 25.0, ramp: float = 0.0):
    """``(linear rgb, depth)`` of a fake ERP: a lit room plus an optional hot band.

    ``ramp`` spreads the radiance over ``[1, 1 + ramp]`` across the frame, which
    a measured display curve needs: a piecewise-constant frame only ever samples
    three radiances.
    """
    import numpy as np

    linear = np.full((height, width, 3), base, dtype="float32")
    depth = np.full((height, width), room_depth, dtype="float32")
    if blown_rows:
        linear[blown_rows[0]:blown_rows[1]] = blown
    if dome_rows:
        linear[:dome_rows] = dome
        depth[:dome_rows] = dome_depth
    if ramp:
        linear = linear * (1.0 + ramp * np.linspace(0.0, 1.0, width, dtype="float32"))[None, :, None]
    return linear, depth


def synthetic_render(root: str, spec: dict, cameras: int = 2, display=None,
                     **frames_kwargs) -> dict:
    """Add a linear-EXR build to ``root``: the inputs ``--fix-exposure`` reads.

    The delivered PNGs are the ``Standard`` display transform of the linear
    data at exposure 0 -- what ``run_build.py --no-denoise`` plus the denoise
    pass write -- while the manifest still claims the render's AgX/exposure.
    ``display`` replaces that colour management with a callable (a foreign view
    transform: the manifest then records it as the render's own).
    """
    import numpy as np

    manifest = manifest_for(spec)
    render = {"width": 64, "height": 32, "samples": 8, "view_transform": "AgX",
              "exposure": 1.4, "post_denoise": {"denoiser": "OPENIMAGEDENOISE",
                                                "prefilter": "ACCURATE", "quality": "FAST"}}
    if display is not None:
        render = {"width": 64, "height": 32, "samples": 8, "view_transform": "AgX",
                  "exposure": 0.0}
    os.makedirs(os.path.join(root, "build"), exist_ok=True)
    entries = []
    for camera in spec["cameras"][:cameras]:
        linear, depth = synthetic_frames(**frames_kwargs)
        raw = f"raw_{camera['name']}0012.exr"
        write_exr(os.path.join(root, "build", raw), linear)
        np.save(os.path.join(root, "build", f"depth_{camera['name']}.npy"), depth)
        view = os.path.join(root, "scene", "viewpoints", camera["name"])
        os.makedirs(view, exist_ok=True)
        if display is None:
            rendered = review_mod._display_rgb(linear, 0.0, "Standard")
        else:
            rendered = np.clip(np.asarray(display(linear), dtype="float32"), 0.0, 1.0)
        png = os.path.join(root, "build", f"pano_{camera['name']}.png")
        Image.fromarray((rendered * 255.0 + 0.5).astype("uint8")).save(png)
        Image.open(png).save(os.path.join(view, "panoImage_2048.png"))
        entries.append({"camera": camera["name"], "raw_exr": raw,
                        "pano": f"pano_{camera['name']}.png", "pos": camera["pos"] + [1.5]})
    manifest["render"] = render
    manifest["cameras"] = entries
    manifest["shell"] = {"dome_radius": 25.0}
    write_json(os.path.join(root, "build", "build_manifest.json"), manifest)
    return manifest


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
        report = review_mod.review(package.root, index=self.index, spec_only=True)
        names = [check["name"] for check in report["checks"]]
        for expected in ("rooms_furnished", "functional_coverage", "no_duplicate_furniture",
                         "camera_clearance", "camera_framing", "camera_coverage"):
            self.assertIn(expected, names)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["pass"], report["issues"])
        self.assertEqual(report["stats"]["issues_fail"], 0)
        self.assertFalse(report["scopes"]["build"], "a spec-only package has no build scope")
        self.assertEqual(report["missing"], [])

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
        package = self.scene(self.spec, panorama=white_blob_panorama, manifest=manifest_for(self.spec))
        report = review_mod.review(package.root, index=self.index)
        self.assertTrue(report["scopes"]["render"])
        exposure = package.check("exposure", report)
        self.assertFalse(exposure["pass"])
        self.assertEqual(exposure["issues"][0]["severity"], "fail")

        dark = self.scene(self.spec, manifest=manifest_for(self.spec),
                          panorama=lambda path: Image.new("RGB", (64, 32), (8, 8, 8)).save(path))
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
        self.assertEqual(report["status"], "error")
        self.assertFalse(report["pass"])

    def test_a_full_review_without_build_or_panoramas_is_not_a_pass(self):
        """Counterexample: a bare spec directory used to review as ``pass: true``.

        ``build_matches_spec``, ``furniture_grounding`` and ``exposure`` have
        nothing to read, so the report is ``not_run`` -- not a green light.
        The callers who asked for less (``--spec-only``, a layout-only run,
        ``--no-images``) say so and are not blocked.
        """
        package = self.scene(self.spec)
        report = review_mod.review(package.root, index=self.index)
        self.assertFalse(report["pass"])
        self.assertEqual(report["status"], "not_run")
        self.assertEqual(sorted(report["missing"]),
                         ["build_matches_spec", "exposure", "furniture_grounding"])
        self.assertEqual(report["scopes_required"], {"spec": True, "build": True, "render": True})
        for name in report["missing"]:
            check = package.check(name, report)
            self.assertEqual(check["status"], "not_run")
            self.assertTrue(check["required"])
            self.assertFalse(check["pass"])

        spec_only = review_mod.review(package.root, index=self.index, spec_only=True)
        self.assertEqual(spec_only["status"], "pass")
        self.assertFalse(spec_only["scopes_required"]["build"])

        layout_only = review_mod.review(package.root, index=self.index, no_build=True)
        self.assertEqual(layout_only["status"], "pass")
        self.assertEqual(layout_only["missing"], [])
        self.assertEqual(package.check("build_matches_spec", layout_only)["required"], False)

        no_images = review_mod.review(package.root, index=self.index, images=False)
        self.assertEqual(no_images["status"], "not_run",
                         "a skipped pixel check does not excuse a missing build")

    def _reviewer(self, package: ScenePackage, payload) -> str:
        """A shell reviewer that prints ``payload`` (junk when it is None)."""
        script = os.path.join(package.root, "reviewer.sh")
        if payload is None:
            body = "#!/usr/bin/env bash\necho not json\nexit 3\n"
        else:
            verdict = os.path.join(package.root, "verdict.json")
            write_json(verdict, payload)
            body = "#!/usr/bin/env bash\ncat '%s'\n" % verdict
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(body)
        os.chmod(script, 0o755)
        return f"bash {script}"

    def test_external_verdict_is_the_reviewers(self):
        package = self.scene(self.spec, panorama=grey_panorama, manifest=manifest_for(self.spec))
        report = review_mod.review(package.root, index=self.index, review_cmd=self._reviewer(
            package, {"pass": True, "score": 0.5, "issues": [], "suggestions": [{"kind": "note"}]}))
        self.assertEqual(report["external"]["status"], "ok")
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["pass"])

        report = review_mod.review(package.root, index=self.index, review_cmd=self._reviewer(
            package, {"pass": False, "issues": [{"detail": "looks wrong", "severity": "fail"}]}))
        self.assertFalse(report["pass"])
        self.assertEqual(report["issues"][-1]["source"], "external")

    def test_external_reject_with_only_warn_notes_still_fails(self):
        """Counterexample: ``pass: false`` plus a default warn note used to stay green."""
        package = self.scene(self.spec, panorama=grey_panorama, manifest=manifest_for(self.spec))
        report = review_mod.review(package.root, index=self.index, review_cmd=self._reviewer(
            package, {"pass": False, "issues": [{"detail": "the sofa is wrong"}]}))
        self.assertFalse(report["pass"])
        self.assertEqual(report["status"], "fail")
        fails = {issue["kind"] for issue in report["issues"] if issue["severity"] == "fail"}
        self.assertIn("external_reject", fails)
        self.assertIn("the sofa is wrong", " ".join(issue.get("detail", "")
                                                    for issue in report["issues"]))

    def test_external_fail_issue_is_not_dropped_by_a_pass_verdict(self):
        """Counterexample: ``pass: true`` with a fail issue lost the fail silently."""
        package = self.scene(self.spec, panorama=grey_panorama, manifest=manifest_for(self.spec))
        report = review_mod.review(package.root, index=self.index, review_cmd=self._reviewer(
            package, {"pass": True, "issues": [{"detail": "the chair floats", "severity": "fail"}]}))
        self.assertFalse(report["pass"])
        self.assertIn("the chair floats", " ".join(issue.get("detail", "")
                                                   for issue in report["issues"]))
        self.assertTrue(report["external"]["verdict_conflict"])
        self.assertIn("external_review",
                      [check["name"] for check in report["checks"] if not check["pass"]])

    def test_external_reviewer_that_cannot_be_reached_is_retryable(self):
        """Counterexample: a crashing hook was recorded and then ignored."""
        package = self.scene(self.spec, panorama=grey_panorama, manifest=manifest_for(self.spec))
        command = self._reviewer(package, None)
        report = review_mod.review(package.root, index=self.index, review_cmd=command)
        self.assertEqual(report["external"]["status"], "error")
        self.assertEqual(report["status"], "error")
        self.assertTrue(report["retry"], "an unreachable reviewer is a retry, not a verdict")
        self.assertFalse(report["pass"])
        check = package.check("external_review", report)
        self.assertTrue(check["required"])
        self.assertTrue(check["details"]["retry"])
        self.assertIn("retry_external_review",
                      [issue["repair"]["kind"] for issue in check["issues"]])

        degraded = review_mod.review(package.root, index=self.index, review_cmd=command,
                                     allow_external_error=True)
        self.assertEqual(degraded["status"], "pass", "the caller may declare the hook optional")
        self.assertTrue(degraded["pass"], degraded["issues"])
        self.assertFalse(package.check("external_review", degraded)["required"])

    def test_review_report_is_written_by_the_cli(self):
        package = self.scene(self.spec)
        report_path = os.path.join(package.root, "cli_report.json")
        result = subprocess.run([HOSTPY, os.path.join(REPO_ROOT, "blender_agent", "review.py"),
                                 package.root, "--spec-only", "--json", report_path],
                                capture_output=True, text=True, cwd=REPO_ROOT)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        report = load_json(report_path)
        self.assertEqual(report["kind"], "blender_agent_review")
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["pass"])
        self.assertIn("review: PASS", result.stdout)

        # the same directory reviewed in full has nothing to read: exit 1, not 0
        full = subprocess.run([HOSTPY, os.path.join(REPO_ROOT, "blender_agent", "review.py"),
                               package.root, "--json", report_path],
                              capture_output=True, text=True, cwd=REPO_ROOT)
        self.assertEqual(full.returncode, 1, full.stderr[-2000:])
        report = load_json(report_path)
        self.assertEqual(report["status"], "not_run")
        self.assertFalse(report["pass"])
        self.assertIn("NOT RUN", full.stdout)


class ExposureTest(unittest.TestCase):
    """The linear-EXR exposure path: statistics, the estimate and the re-do."""

    @unittest.skipUnless(HAVE_IMAGES, "numpy/PIL not available")
    def test_exr_reader_reads_what_the_writer_wrote(self):
        import numpy as np

        root = tempfile.mkdtemp(prefix="exr_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        rgb = np.zeros((32, 64, 3), dtype="float32")
        rgb[..., 0] = np.linspace(0.0, 3.0, 64)[None, :]
        rgb[..., 1] = np.linspace(0.0, 1.5, 32)[:, None]
        rgb[..., 2] = 0.25
        path = os.path.join(root, "small.exr")
        write_exr(path, rgb)
        read = review_mod.read_exr(path)
        self.assertEqual(read["compression"], "ZIP")
        self.assertEqual(read["size"], (64, 32))
        for index, name in enumerate(("R", "G", "B")):
            np.testing.assert_allclose(read["channels"][name], rgb[..., index], rtol=0, atol=1e-6)

    @unittest.skipUnless(HAVE_IMAGES, "numpy/PIL not available")
    def test_erp_weights_follow_the_latitude(self):
        import numpy as np

        weights = review_mod._erp_weights(64)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertLess(float(weights[0]), 0.1, "the pole covers no solid angle")
        self.assertGreater(float(weights[32]), 0.99, "the horizon has the full weight")
        polar = np.zeros((64, 16), dtype=bool)
        polar[:4] = True
        middle = np.zeros((64, 16), dtype=bool)
        middle[30:34] = True
        self.assertLess(review_mod._weighted_share(polar, weights),
                        review_mod._weighted_share(middle, weights))

    def _package(self, spec: dict = None, **frames) -> ScenePackage:
        spec = spec or cached_spec()
        package = ScenePackage(spec)
        self.addCleanup(shutil.rmtree, package.root, ignore_errors=True)
        synthetic_render(package.root, spec, **frames)
        return package

    @unittest.skipUnless(HAVE_IMAGES, "numpy/PIL not available")
    def test_a_panorama_that_clips_on_the_dome_is_a_warning_not_a_gate(self):
        """A bright sky above the walls is quality headroom, not a broken render."""
        package = self._package(dome_rows=12, dome=2.5, blown_rows=None)
        self.index = checks_mod.load_assets_index()
        report = review_mod.review(package.root, index=self.index)
        exposure = package.check("exposure", report)
        self.assertEqual({issue["severity"] for issue in exposure["issues"]}, {"warn"})
        kinds = {issue["kind"] for issue in exposure["issues"]}
        self.assertIn("clipped_window", kinds)
        self.assertNotIn("blown_out_view", kinds)
        self.assertEqual(report["status"], "pass", report["issues"])
        detail = exposure["details"]["cameras"]["0000"]
        self.assertGreater(detail["exterior_clip_fraction"], 0.15)
        self.assertEqual(detail["interior_clip_fraction"], 0.0)
        self.assertEqual(detail["clipping"], "interior+exterior")
        estimate = review_mod.estimate_exposure(
            review_mod.Artifacts(package.root), index=self.index)
        self.assertEqual(estimate["status"], "nothing_to_do")

    @unittest.skipUnless(HAVE_IMAGES, "numpy/PIL not available")
    def test_a_blown_wall_fails_and_the_estimate_measures_the_drop(self):
        package = self._package(blown_rows=(11, 24), blown=2.5)
        self.index = checks_mod.load_assets_index()
        report = review_mod.review(package.root, index=self.index)
        fails = [issue for issue in report["issues"] if issue["severity"] == "fail"]
        self.assertEqual({issue["kind"] for issue in fails}, {"blown_out_view"})
        self.assertEqual(report["status"], "fail")
        estimate = review_mod.estimate_exposure(
            review_mod.Artifacts(package.root), index=self.index)
        self.assertEqual(estimate["status"], "ok")
        self.assertLess(estimate["delta_stops"], -1.0)
        self.assertEqual(estimate["failing_cameras"], ["0000", "0001"])
        self.assertEqual(len(set(estimate["predicted"])), 2,
                         "one exposure is chosen for the whole viewpoint set")
        for camera in estimate["predicted"].values():
            self.assertLess(camera["interior_clip_fraction"], review_mod.CLIP_FAIL_FRACTION)
            self.assertLess(camera["interior_blob_fraction"], review_mod.CLIP_BLOB_FAIL)
            self.assertGreater(camera["mean_luminance"],
                               review_mod.DARK_WARN_LUMINANCE,
                               "the fix must not drown the interior")

    @unittest.skipUnless(HAVE_IMAGES, "numpy/PIL not available")
    def test_fix_exposure_rewrites_the_panoramas_without_a_render(self):
        package = self._package(blown_rows=(11, 24), blown=2.5)
        index = checks_mod.load_assets_index()
        spec_before = load_json(os.path.join(package.root, "spec.json"))
        before = read_png(os.path.join(package.root, "build", "pano_0000.png"))
        result = review_mod.fix_exposure(package.root, index=index, use_blender=False,
                                         review_cmd=None)
        self.assertEqual(result["status"], "applied", result)
        self.assertTrue(result["changed"])
        self.assertLess(result["exposure_after"], 0.0)
        self.assertTrue(result["review"]["pass"], result["review"])
        after = read_png(os.path.join(package.root, "build", "pano_0000.png"))
        self.assertEqual(float((pixel_luminance(after) >= 0.98).mean()), 0.0)
        self.assertLess(float(pixel_luminance(after).mean()),
                        float(pixel_luminance(before).mean()))
        # build panorama and exported viewpoint stay the same image
        import numpy as np

        view = os.path.join(package.root, "scene", "viewpoints", "0000", "panoImage_2048.png")
        np.testing.assert_array_equal(np.asarray(Image.open(view)),
                                      np.asarray(Image.open(os.path.join(
                                          package.root, "build", "pano_0000.png"))))
        # the geometry spec is untouched: the fix is a colour-management redo
        self.assertEqual(load_json(os.path.join(package.root, "spec.json")), spec_before)
        sidecar = load_json(os.path.join(package.root, "review_exposure.json"))
        self.assertEqual(sidecar["status"], "applied")
        self.assertEqual(sidecar["cameras"]["0000"]["exposure"], result["exposure_after"])
        self.assertTrue(os.path.isfile(os.path.join(package.root, "build", "pano_pre_exposure",
                                                    "pano_0000.png")))
        # a second run finds nothing left to do
        again = review_mod.fix_exposure(package.root, index=index, use_blender=False,
                                        review_cmd=None)
        self.assertEqual(again["status"], "nothing_to_do")

    @unittest.skipUnless(HAVE_IMAGES, "numpy/PIL not available")
    def test_measured_curve_follows_a_transform_we_cannot_reproduce(self):
        import numpy as np

        generator = np.random.default_rng(7)
        x = generator.uniform(0.02, 4.0, 40000)
        y = 1.0 - np.exp(-x)                       # a filmic-ish, monotone transform
        curve = review_mod._measured_curve(x, y)
        self.assertIsNotNone(curve)
        for shift in (0.0, -0.5, -1.0):
            predicted = review_mod._apply_curve(x * (2.0 ** shift), curve)
            truth = 1.0 - np.exp(-(x * (2.0 ** shift)))
            self.assertLess(float(np.mean(np.abs(predicted - truth))), 0.03)

    @unittest.skipUnless(HAVE_IMAGES, "numpy/PIL not available")
    def test_a_foreign_view_transform_is_measured_from_the_delivered_panoramas(self):
        """AgX and friends: the estimate reads the response off the images."""
        import numpy as np

        def filmic(linear):
            return 1.0 - np.exp(-np.asarray(linear, dtype="float32"))

        package = self._package(display=filmic, blown_rows=(11, 24), blown=6.0, ramp=0.6)
        index = checks_mod.load_assets_index()
        report = review_mod.review(package.root, index=index)
        self.assertEqual(report["status"], "fail", report["issues"])
        estimate = review_mod.estimate_exposure(review_mod.Artifacts(package.root), index=index)
        self.assertEqual(estimate["status"], "ok")
        self.assertEqual(estimate["transform"], "AgX")
        self.assertTrue(any("measured" in warning for warning in estimate["warnings"]))
        for camera in estimate["predicted"].values():
            self.assertLess(camera["interior_clip_fraction"], review_mod.CLIP_FAIL_FRACTION)
        # the rewrite itself needs the real transform, which only Blender has
        result = review_mod.fix_exposure(package.root, index=index, use_blender=False)
        self.assertEqual(result["status"], "unsupported_transform")
        self.assertFalse(result["changed"])

    @unittest.skipUnless(HAVE_IMAGES, "numpy/PIL not available")
    def test_apply_repair_uses_the_measured_exposure(self):
        package = self._package(blown_rows=(11, 24), blown=2.5)
        index = checks_mod.load_assets_index()
        report = review_mod.review(package.root, index=index)
        result = review_mod.apply_repair(package.root, report=report, index=index)
        self.assertTrue(result["changed"], result)
        actions = [action for action in result["actions"] if action["kind"] == "reduce_exposure"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["measured"], "linear EXR statistics")
        self.assertLess(actions[0]["exposure_after"], actions[0]["exposure_before"])
        repaired = load_json(os.path.join(package.root, "spec.json"))
        self.assertEqual(repaired["style"]["exposure"], actions[0]["exposure_after"])

    @unittest.skipUnless(HAVE_IMAGES, "numpy/PIL not available")
    @unittest.skipUnless(os.environ.get("PANOWORLD_BLENDER_TEST") == "1",
                         "set PANOWORLD_BLENDER_TEST=1 to exercise the Blender re-do")
    def test_blender_redo_reproduces_the_same_pixels(self):
        package = self._package(blown_rows=(11, 24), blown=2.5)
        index = checks_mod.load_assets_index()
        estimate = review_mod.estimate_exposure(review_mod.Artifacts(package.root), index=index)
        result = review_mod.fix_exposure(package.root, index=index, use_blender=True,
                                         review_after=False)
        self.assertEqual(result["status"], "applied", result)
        self.assertTrue(result["cameras"]["0000"]["denoised"],
                        "the Blender pass denoises, it does not fall back to the sRGB rewrite")
        blender = read_png(os.path.join(package.root, "build", "pano_0000.png"))
        source = review_mod.read_exr(os.path.join(package.root, "build",
                                                  "raw_00000012.exr"))
        expected = review_mod._display_rgb(
            __import__("numpy").stack([source["channels"][name] for name in ("R", "G", "B")], -1),
            estimate["exposure_after"], "Standard")
        self.assertLess(float(abs(blender - expected).mean()), 0.01)


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
