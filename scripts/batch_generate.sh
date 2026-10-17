#!/usr/bin/env bash
# Batch layout generation: one directory per seed plus an aggregate table.
#
# 用法: bash scripts/batch_generate.sh <start_seed> <count> <out_root>
#
# Layout: <out_root>/scene_<seed>/{spec.json, layout_report.json,
#                                   checks_report.json, run_summary.json,
#                                   resume_manifest.json, validation_report.json,
#                                   run.log, ...}
#         <out_root>/batch_summary.json
#
# batch_summary.json (schema_version 2) grades every scene on four levels --
# layout_ok / render_ok / validation_ok / bundle_ok -- instead of folding
# layout-only runs and memory-skipped renders into one pass count.  With
# PANOWORLD_REVIEW=1 the child runs also review every scene and the levels gain
# a fifth line, review_ok.  "passed" counts bundle_ok only: a render that was
# requested but did not complete is pending_retry (or failed), never a pass;
# failure_reasons / failed_checks / review_issues carry the distribution.  A
# failing scene never stops the batch, but the script exits 0 only when every
# scene reached bundle_ok (1 otherwise).
#
# Resume is request-bound, not summary-bound.  Every scene the batch runs gets
# a <scene_dir>/resume_manifest.json recording the *request* (seed, resolution,
# samples, floors, shape, rooms, review mode), the fingerprints of the code and
# of the asset registry that produced it, the effective render profile (which
# differs from the requested one when the memory retry reduced it), and the
# normalized spec hash.  PANOWORLD_BATCH_RESUME=1 reuses a scene only when that
# manifest parses, the current request hashes to the recorded one, spec.json
# still hashes to the recorded value, run_summary.json is a complete bundle of
# the requested mode, and every camera of spec.json still has a decodable
# panorama, depth map, depth scale and pose whose size matches the effective
# profile.  Anything else is re-run; a scene without a manifest (produced by a
# batch older than this contract) is never reused, because its request cannot
# be proven to match.
#
# A run whose memory retry succeeded at a reduced profile is graded bundle_ok
# but reported as *degraded*: row["render_quality"] carries the requested and
# the effective profile, batch_summary.json counts them separately
# ("degraded" / "degraded_seeds"), and the table prints the reduced profile, so
# a quality-downgraded scene can never be read as a full-quality completion.
#
# Every row also carries a "family" block -- the arrangement the scene was
# built as, the family the seed drew (stats.auto_pick), whether the sticky auto
# draw had to fall back and how many attempts it took -- and the summary counts
# the built mix ("families").  Without it a reader can only count what the
# batch *asked* for, which is exactly how a sweep that keeps failing on the
# hard families looks diverse while delivering row after row; see
# `tools/diversity_report.py success`.
#
# Environment:
#   PANOWORLD_BATCH_RENDER=1        also run the Blender pipeline for every
#                                   scene (default 0: layout + checks only,
#                                   so a 20 seed sweep is seconds long)
#   PANOWORLD_BATCH_RESUME=1        keep scenes whose resume_manifest.json
#                                   proves the request matches and whose
#                                   artifacts are complete instead of
#                                   re-running them (interrupted batch)
#   PANOWORLD_BATCH_SUMMARY_ONLY=1  rebuild batch_summary.json from the
#                                   existing scene dirs and run nothing
#   PANOWORLD_BATCH_SEEDS=0,7,13    explicit seed list (START/COUNT ignored)
#   PANOWORLD_BATCH_ID              batch id (default: timestamp + pid)
#   PANOWORLD_REVIEW=1              forward the visual/rule review to the
#                                   per-scene runs and grade review_ok
#   PANOWORLD_REVIEW_REPAIR=1       let a failing review patch its spec once
#   PANOWORLD_REVIEW_STRICT=1       warnings fail the review as well
#   PANOWORLD_RES / PANOWORLD_SAMPLES / PANOWORLD_MIN_FREE_MB
#                                   forwarded to layout_gen_run.sh; when unset
#                                   they resolve from PANOWORLD_LAYOUT_CONFIG
#                                   and are then pinned for every child run, so
#                                   the request recorded in the manifest is the
#                                   profile the scenes were really rendered at
#   PANOWORLD_ROOMS                 room count (layout_gen.py decides its range)
#   PANOWORLD_LAYOUT_FLOORS         1 (default) or 2: duplex scenes
#   PANOWORLD_LAYOUT_SHAPE          room arrangement (row/block/auto/<family>)
#   PANOWORLD_LAYOUT_CONFIG         generator config (render defaults)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOSTPY="${PANOWORLD_HOSTPY:-/opt/conda/envs/diffaero/bin/python}"
START="${1:?usage: batch_generate.sh <start_seed> <count> <out_root>}"
COUNT="${2:?usage: batch_generate.sh <start_seed> <count> <out_root>}"
ROOT="${3:?usage: batch_generate.sh <start_seed> <count> <out_root>}"
RENDER="${PANOWORLD_BATCH_RENDER:-0}"
ROOMS="${PANOWORLD_ROOMS:-}"
SEED_LIST="${PANOWORLD_BATCH_SEEDS:-}"
RESUME="${PANOWORLD_BATCH_RESUME:-0}"
SUMMARY_ONLY="${PANOWORLD_BATCH_SUMMARY_ONLY:-0}"
REVIEW="${PANOWORLD_REVIEW:-0}"
REVIEW_REPAIR="${PANOWORLD_REVIEW_REPAIR:-0}"
REVIEW_STRICT="${PANOWORLD_REVIEW_STRICT:-0}"
LAYOUT_CONFIG="${PANOWORLD_LAYOUT_CONFIG:-${REPO}/configs/layout_gen_demo.json}"
LAYOUT_FLOORS="${PANOWORLD_LAYOUT_FLOORS:-1}"
LAYOUT_SHAPE="${PANOWORLD_LAYOUT_SHAPE:-row}"
BATCH_ID="${PANOWORLD_BATCH_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# The render defaults of layout_gen_run.sh come from the config file unless the
# environment overrides them; the batch resolves the same values here and
# exports them, so the profile recorded in the manifest is the one the child
# runs at (the helper prints an empty line when the key or file is absent).
config_value() { # <dotted.key>
  [[ -r "${LAYOUT_CONFIG}" ]] || return 0
  "${HOSTPY}" -c 'import json,sys
try:
    node = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print(""); raise SystemExit
for key in sys.argv[2].split("."):
    node = node.get(key, {}) if isinstance(node, dict) else {}
print(node if not isinstance(node, dict) else "")' "${LAYOUT_CONFIG}" "$1" 2>/dev/null || true
}

RES="${PANOWORLD_RES:-$(config_value render.resolution)}"
SAMPLES="${PANOWORLD_SAMPLES:-$(config_value render.samples)}"
RES="${RES:-1024x512}"
SAMPLES="${SAMPLES:-32}"
export PANOWORLD_RES="${RES}" PANOWORLD_SAMPLES="${SAMPLES}"
export PANOWORLD_LAYOUT_FLOORS="${LAYOUT_FLOORS}" PANOWORLD_LAYOUT_SHAPE="${LAYOUT_SHAPE}"

WORK_DIR=""
cleanup() {
  [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]] && rm -rf "${WORK_DIR}"
  return 0
}
trap cleanup EXIT
BATCH_REPO="${REPO}"
export BATCH_REPO

mkdir -p "${ROOT}"
ROOT="$(cd "${ROOT}" && pwd)"
echo "== batch_generate =="
echo "   batch id: ${BATCH_ID}"
echo "   seeds   : ${START} .. $(( START + COUNT - 1 ))"
echo "   out     : ${ROOT}"
echo "   render  : ${RENDER} (resume=${RESUME}, summary_only=${SUMMARY_ONLY}) at ${RES}, ${SAMPLES} samples"
echo "   layout  : floors=${LAYOUT_FLOORS} shape=${LAYOUT_SHAPE} rooms=${ROOMS:-<seed>}"
echo "   review  : ${REVIEW} (repair=${REVIEW_REPAIR}, strict=${REVIEW_STRICT})"

SEEDS=()
if [[ -n "${SEED_LIST}" ]]; then
  IFS=',' read -r -a SEEDS <<< "${SEED_LIST}"
else
  for (( offset = 0; offset < COUNT; offset++ )); do
    SEEDS+=("$(( START + offset ))")
  done
fi

RESUMED_SEEDS=()

# --------------------------------------------------------------------------- #
# resume manifest tool
#
# One embedded program drives every fingerprint decision of the batch:
#
#   template <path>                      write the seed-less request block
#   decision <scene_dir> <tpl> <seed>    print "resume<TAB>detail" or
#                                        "rerun<TAB>reason"
#   manifest <scene_dir> <tpl> <seed> <batch_id> <run_id>
#                                        write resume_manifest.json
#
# The request block carries what a scene was asked for (seed, resolution,
# samples, floors, shape, rooms, review mode), the sha256 of every
# generation/render/validation source file, the git commit for provenance (it
# is not compared: the content hashes already are), and the asset registry the
# generator reads.  The gate digests the block without the git record, so
# committing the same tree keeps caches valid while a real edit invalidates
# them.  The manifest records the *effective* render profile as well, which is
# the reduced one after a memory retry succeeded.
# --------------------------------------------------------------------------- #
batch_tool() { # <mode> [args...]
  "${HOSTPY}" - "$@" <<'PY'
"""Batch-side request fingerprint, resume decision and scene manifest."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.environ.get("BATCH_REPO") or os.getcwd()
MANIFEST_NAME = "resume_manifest.json"
MANIFEST_KIND = "batch_scene_manifest"
SCHEMA_VERSION = 1

#: Sources whose contents decide the spec and the artifacts derived from it.
#: A bundle produced by another revision of any of these is never reused.
CODE_FILES = (
    "blender_agent/layout_gen.py",
    "blender_agent/spec.py",
    "blender_agent/checks.py",
    "blender_agent/fixtures.py",
    "blender_agent/assets.py",
    "blender_agent/shell.py",
    "blender_agent/materials.py",
    "blender_agent/details.py",
    "blender_agent/settle.py",
    "blender_agent/run_build.py",
    "blender_agent/panorama.py",
    "blender_agent/export_scene.py",
    "blender_agent/review.py",
    "scripts/layout_gen_run.sh",
    "scripts/blender_agent_run.sh",
    "scripts/batch_generate.sh",
    "tools/validate_blender_agent_p0.py",
    "configs/layout_gen_demo.json",
)
#: Asset registries the generator sizes its furniture from (merged index first,
#: the Poly Haven cache otherwise -- the same order as checks.DEFAULT_ASSETS_INDEX).
ASSET_INDEX_CANDIDATES = (
    os.path.join("third_party", "assets", "merged_index.json"),
    os.path.join("third_party", "assets", "polyhaven", "index.json"),
)
#: Files a bundle_ok render of a scene must still have on disk.
REQUIRED_FILES = ("spec.json", "checks_report.json")
VIEWPOINT_FILES = ("panoImage_2048.png", "place_depth.png", "place_depth_scale.txt",
                   "extrinsics.txt")


# ------------------------------------------------------------------------- #
# hashing
# ------------------------------------------------------------------------- #
def canonical(node) -> str:
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def digest(node) -> str:
    return hashlib.sha256(canonical(node).encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    handle = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            handle.update(chunk)
    return handle.hexdigest()


def repo_path(path: str) -> str:
    """Paths inside the repo are recorded relative, so a moved checkout or a
    frozen copy of the same tree hashes the same."""
    try:
        relative = os.path.relpath(os.path.abspath(path), REPO)
    except ValueError:
        return os.path.abspath(path)
    return path if relative.startswith("..") else relative


def normalize(value):
    """A spec normalized for hashing: key order and float formatting dropped."""
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return repr(value)
        rounded = round(value, 9)
        return 0.0 if rounded == 0 else rounded
    return value


def normalize_spec_hash(spec) -> str:
    return digest(normalize(spec))


def read_json(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


# ------------------------------------------------------------------------- #
# request / fingerprints
# ------------------------------------------------------------------------- #
def code_block() -> dict:
    files = {}
    for name in CODE_FILES:
        path = os.path.join(REPO, name)
        files[name] = sha256_file(path) if os.path.isfile(path) else None
    return {"sha256": digest(files), "files": files, "git": git_provenance()}


def git_provenance() -> dict:
    def run(args):
        try:
            result = subprocess.run(["git", "-C", REPO] + args, capture_output=True,
                                    text=True, timeout=20)
        except Exception:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = run(["status", "--porcelain"])
    return {"commit": run(["rev-parse", "HEAD"]),
            "dirty": None if status is None else bool(status)}


def assets_block() -> dict:
    index = next((os.path.join(REPO, name) for name in ASSET_INDEX_CANDIDATES
                  if os.path.isfile(os.path.join(REPO, name))), None)
    return {"index": repo_path(index) if index else None,
            "sha256": sha256_file(index) if index else None}


def request_block() -> dict:
    """The request as the environment describes it (the seed is added per scene)."""
    config = os.environ.get("PANOWORLD_LAYOUT_CONFIG") or os.path.join(
        REPO, "configs", "layout_gen_demo.json")
    rooms = os.environ.get("PANOWORLD_ROOMS") or ""
    return {
        "render": {"requested": os.environ.get("BATCH_RENDER") == "1",
                   "resolution": os.environ.get("PANOWORLD_RES") or "1024x512",
                   "samples": int(os.environ.get("PANOWORLD_SAMPLES") or 32)},
        "review": {"requested": os.environ.get("BATCH_REVIEW") == "1",
                   "repair": os.environ.get("BATCH_REVIEW_REPAIR") == "1",
                   "strict": os.environ.get("BATCH_REVIEW_STRICT") == "1"},
        "layout": {"floors": os.environ.get("PANOWORLD_LAYOUT_FLOORS") or "1",
                   "shape": os.environ.get("PANOWORLD_LAYOUT_SHAPE") or "row",
                   "rooms": int(rooms) if rooms else None},
        "config": {"path": repo_path(config),
                   "sha256": sha256_file(config) if os.path.isfile(config) else None},
        "assets": assets_block(),
        "code": code_block(),
    }


def gate_block(request: dict) -> dict:
    """The part of a request the resume decision compares (git is provenance)."""
    gate = {key: value for key, value in request.items() if key != "code"}
    gate["code"] = {key: value for key, value in (request.get("code") or {}).items()
                    if key != "git"}
    return gate


def request_for(template_path: str, seed: int) -> dict:
    template = read_json(template_path)
    if not isinstance(template, dict):
        raise SystemExit("batch_tool: unreadable request template %s" % template_path)
    request = json.loads(json.dumps(template))
    request["seed"] = int(seed)
    return request


def flatten(node, prefix: str = "") -> dict:
    out = {}
    if isinstance(node, dict):
        for key in sorted(node):
            out.update(flatten(node[key], "%s.%s" % (prefix, key) if prefix else str(key)))
    elif isinstance(node, list):
        out[prefix] = json.dumps(node)
    else:
        out[prefix] = node
    return out


def describe_change(old, new, limit: int = 3) -> str:
    left, right = flatten(old), flatten(new)

    def short(value):
        text = repr(value)
        return text if len(text) <= 24 else text[:20] + "...'"

    parts = ["%s: %s -> %s" % (key, short(left.get(key)), short(right.get(key)))
             for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
    if not parts:
        return "no field changed (digest mismatch)"
    if len(parts) > limit:
        parts = parts[:limit] + ["..."]
    return "; ".join(parts)


# ------------------------------------------------------------------------- #
# scene facts
# ------------------------------------------------------------------------- #
def effective_profile(summary) -> dict:
    """The profile of the render that produced the artifacts on disk."""
    render = (summary or {}).get("render") or {}
    if render.get("status") not in ("ok", "validation_flagged"):
        return None
    attempts = render.get("attempts") or []
    if not attempts:
        return None
    last = attempts[-1] or {}
    if not last.get("resolution"):
        return None
    return {"resolution": str(last["resolution"]),
            "samples": None if last.get("samples") is None else int(last["samples"]),
            "attempt": last.get("attempt"),
            "status": last.get("status")}


def requested_profile(summary) -> dict:
    render = (summary or {}).get("render") or {}
    return {"resolution": render.get("resolution"),
            "samples": None if render.get("samples") is None else int(render["samples"])}


def render_block(summary, request) -> dict:
    requested = requested_profile(summary)
    effective = effective_profile(summary)
    degraded = bool(effective and request["render"]["requested"] and requested["resolution"]
                    and (effective["resolution"] != requested["resolution"]
                         or (effective["samples"] is not None
                             and requested["samples"] is not None
                             and effective["samples"] != requested["samples"])))
    render = (summary or {}).get("render") or {}
    return {"status": render.get("status"),
            "requested_profile": requested,
            "effective_profile": effective,
            "degraded": degraded,
            "attempts": len(render.get("attempts") or [])}


def spec_block(scene_dir: str):
    spec_path = os.path.join(scene_dir, "spec.json")
    if not os.path.isfile(spec_path):
        return None
    spec = read_json(spec_path)
    if not isinstance(spec, dict):
        return {"present": True, "unreadable": True, "sha256": None}
    report = read_json(os.path.join(scene_dir, "layout_report.json")) or {}
    stats = report.get("stats") or {}
    return {
        "present": True,
        "sha256": normalize_spec_hash(spec),
        "bytes": os.path.getsize(spec_path),
        "floors": len(spec.get("floors") or []) or 1,
        "rooms": len(spec.get("rooms") or []),
        "room_types": [room.get("type") for room in spec.get("rooms") or []],
        "cameras": [camera.get("name") for camera in spec.get("cameras") or []],
        "openings": len(spec.get("openings") or []),
        "connectors": len(spec.get("connectors") or []),
        "shape": report.get("shape") or stats.get("shape"),
        "family": report.get("family"),
    }


def size_of(resolution):
    if not resolution or "x" not in str(resolution):
        return None
    width, _, height = str(resolution).partition("x")
    try:
        return int(width), int(height)
    except ValueError:
        return None


def artifact_problems(scene_dir: str, spec, expected_size) -> list:
    """Every camera must still hold a decodable panorama, depth map, scale and
    pose of the expected size -- the same contract validate_blender_agent_p0.py
    grades, re-checked here because files can disappear after a passed report."""
    problems = []
    if not isinstance(spec, dict):
        return ["spec.json missing or unreadable"]
    cameras = [str(camera.get("name")) for camera in spec.get("cameras") or []]
    if not cameras:
        return ["spec.json lists no camera"]
    if expected_size is None:
        return ["cannot tell which resolution the artifacts should have"]
    width, height = expected_size
    scene = os.path.join(scene_dir, "scene")
    for name in ("map.json", "topdown.png"):
        if not os.path.isfile(os.path.join(scene, name)):
            problems.append("scene/%s missing" % name)
    try:
        import numpy as np
        from PIL import Image
    except Exception as error:                       # pragma: no cover
        return problems + ["cannot decode the artifacts (%s)" % error]
    for name in cameras:
        view = os.path.join(scene, "viewpoints", name)
        if not os.path.isdir(view):
            problems.append("scene/viewpoints/%s missing" % name)
            continue
        for filename in VIEWPOINT_FILES:
            if not os.path.isfile(os.path.join(view, filename)):
                problems.append("scene/viewpoints/%s/%s missing" % (name, filename))
        pano_path = os.path.join(view, "panoImage_2048.png")
        if os.path.isfile(pano_path):
            try:
                with Image.open(pano_path) as pano:
                    if pano.size != (width, height) or pano.mode not in ("RGB", "RGBA"):
                        problems.append("scene/viewpoints/%s/panoImage_2048.png is %s %s, "
                                        "expected %s at %sx%s"
                                        % (name, pano.size, pano.mode, (width, height),
                                           width, height))
                    pano.load()
            except Exception as error:
                problems.append("scene/viewpoints/%s/panoImage_2048.png undecodable (%s)"
                                % (name, error))
        depth_path = os.path.join(view, "place_depth.png")
        scale_path = os.path.join(view, "place_depth_scale.txt")
        if os.path.isfile(depth_path):
            try:
                with Image.open(depth_path) as depth_image:
                    raw = np.array(depth_image)
                if raw.dtype != np.uint16 or raw.shape != (height, width):
                    problems.append("scene/viewpoints/%s/place_depth.png is %s %s, expected "
                                    "uint16 (%s, %s)"
                                    % (name, raw.dtype, raw.shape, height, width))
            except Exception as error:
                problems.append("scene/viewpoints/%s/place_depth.png undecodable (%s)"
                                % (name, error))
        if os.path.isfile(scale_path):
            try:
                with open(scale_path, encoding="utf-8") as handle:
                    if float(handle.read().strip()) <= 0:
                        problems.append("scene/viewpoints/%s/place_depth_scale.txt is not "
                                        "positive" % name)
            except Exception as error:
                problems.append("scene/viewpoints/%s/place_depth_scale.txt unreadable (%s)"
                                % (name, error))
        pose_path = os.path.join(view, "extrinsics.txt")
        if os.path.isfile(pose_path):
            try:
                matrix = np.loadtxt(pose_path)
                rotation = matrix[:3, :3]
                determinant = float(np.linalg.det(rotation))
                orthogonal = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all() \
                        or abs(determinant - 1.0) > 1e-6 or orthogonal > 1e-6:
                    problems.append("scene/viewpoints/%s/extrinsics.txt is not a rigid "
                                    "pose (shape=%s, det=%.6f, orthogonality=%.2e)"
                                    % (name, list(matrix.shape), determinant, orthogonal))
            except Exception as error:
                problems.append("scene/viewpoints/%s/extrinsics.txt unreadable (%s)"
                                % (name, error))
    return problems


def expected_size_for(summary, manifest=None):
    effective = effective_profile(summary)
    if effective is not None:
        size = size_of(effective.get("resolution"))
        if size:
            return size
    if isinstance(manifest, dict):
        recorded = ((manifest.get("render") or {}).get("effective_profile") or {})
        size = size_of(recorded.get("resolution"))
        if size:
            return size
    return size_of(requested_profile(summary).get("resolution"))


# ------------------------------------------------------------------------- #
# modes
# ------------------------------------------------------------------------- #
def mode_template(path: str) -> int:
    request = request_block()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("request template: code %s, assets %s, %s/%s"
          % (request["code"]["sha256"][:16],
             (request["assets"]["sha256"] or "none")[:16],
             request["render"]["resolution"], request["render"]["samples"]))
    return 0


def mode_decision(scene_dir: str, template_path: str, seed: int) -> int:
    request = request_for(template_path, seed)
    render_expected = request["render"]["requested"]
    review_expected = request["review"]["requested"]
    manifest = read_json(os.path.join(scene_dir, MANIFEST_NAME))
    if not isinstance(manifest, dict):
        return verdict("rerun", "no %s: a run from before the fingerprinting batch "
                                "cannot prove the request matches" % MANIFEST_NAME)
    if manifest.get("kind") != MANIFEST_KIND or manifest.get("schema_version") != SCHEMA_VERSION:
        return verdict("rerun", "manifest is not %s schema %s" % (MANIFEST_KIND, SCHEMA_VERSION))
    if manifest.get("request_sha256") != digest(gate_block(request)):
        return verdict("rerun", "request changed: "
                       + describe_change(gate_block(manifest.get("request") or {}),
                                         gate_block(request)))

    summary = read_json(os.path.join(scene_dir, "run_summary.json"))
    if not isinstance(summary, dict):
        return verdict("rerun", "no readable run_summary.json")
    ok = summary.get("ok") or {}
    if not ok.get("bundle_ok"):
        return verdict("rerun", "run_summary.json is not a complete bundle")
    if bool(summary.get("render_requested")) != render_expected:
        return verdict("rerun", "render_requested=%s in the summary, %s requested"
                       % (summary.get("render_requested"), render_expected))
    if review_expected and ok.get("review_ok") is not True:
        return verdict("rerun", "review_ok is not true")
    needed = list(REQUIRED_FILES)
    if render_expected:
        needed.append("validation_report.json")
    if review_expected:
        needed.append("review_report.json")
    missing = [name for name in needed if not os.path.isfile(os.path.join(scene_dir, name))]
    if missing:
        return verdict("rerun", "missing " + ", ".join(missing))

    spec = read_json(os.path.join(scene_dir, "spec.json"))
    if not isinstance(spec, dict):
        return verdict("rerun", "no readable spec.json")
    recorded_spec = (manifest.get("spec") or {}).get("sha256")
    if not recorded_spec:
        return verdict("rerun", "manifest records no spec hash")
    if normalize_spec_hash(spec) != recorded_spec:
        return verdict("rerun", "spec.json changed since the run (hash mismatch)")

    if render_expected:
        problems = artifact_problems(scene_dir, spec, expected_size_for(summary, manifest))
        if problems:
            return verdict("rerun", "artifacts incomplete: " + summarize(problems))
    return verdict("resume", detail_of(manifest, summary))


def summarize(problems, limit: int = 4) -> str:
    if len(problems) <= limit:
        return "; ".join(problems)
    return "; ".join(problems[:limit]) + " (+%d more)" % (len(problems) - limit)


def detail_of(manifest, summary) -> str:
    request = manifest.get("request") or {}
    render = manifest.get("render") or {}
    parts = ["run %s" % (manifest.get("run_id") or "?"),
             "%s, %s samples" % (request.get("render", {}).get("resolution"),
                                 request.get("render", {}).get("samples"))]
    effective = render.get("effective_profile")
    if render.get("degraded") and effective:
        parts.append("degraded to %s, %s samples" % (effective.get("resolution"),
                                                     effective.get("samples")))
    spec = manifest.get("spec") or {}
    if spec.get("present"):
        parts.append("%s floors, %s rooms, %s cameras"
                     % (spec.get("floors"), spec.get("rooms"), len(spec.get("cameras") or [])))
    return ", ".join(str(part) for part in parts)


def verdict(word: str, detail: str) -> int:
    print("%s\t%s" % (word, detail))
    return 0


def mode_manifest(scene_dir: str, template_path: str, seed: int,
                  batch_id: str, run_id: str) -> int:
    request = request_for(template_path, seed)
    summary = read_json(os.path.join(scene_dir, "run_summary.json")) or {}
    spec = read_json(os.path.join(scene_dir, "spec.json"))
    render = render_block(summary, request)
    problems = []
    if request["render"]["requested"] and render["status"] == "ok":
        problems = artifact_problems(scene_dir, spec, expected_size_for(summary))
    manifest = {
        "kind": MANIFEST_KIND,
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "batch_id": batch_id,
        "run_id": run_id,
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "request": request,
        "request_sha256": digest(gate_block(request)),
        "spec": spec_block(scene_dir),
        "render": render,
        "artifacts": {"viewpoints": [camera.get("name")
                                     for camera in (spec or {}).get("cameras") or []],
                      "problems": problems},
        "bundle_ok": bool((summary.get("ok") or {}).get("bundle_ok")),
        "exit_code": summary.get("exit_code"),
    }
    path = os.path.join(scene_dir, MANIFEST_NAME)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return 0


def main(argv) -> int:
    if not argv:
        sys.stderr.write("usage: batch_tool <template|decision|manifest> ...\n")
        return 2
    mode = argv[0]
    if mode == "template":
        return mode_template(argv[1])
    if mode == "decision":
        return mode_decision(argv[1], argv[2], int(argv[3]))
    if mode == "manifest":
        return mode_manifest(argv[1], argv[2], int(argv[3]), argv[4], argv[5])
    sys.stderr.write("batch_tool: unknown mode %r\n" % mode)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
PY
}

# Prints "<verdict>\t<detail>": "resume" when the manifest proves the request
# matches and the artifacts are complete, "rerun" with the reason otherwise.
resume_decision() { # <seed>
  local seed="$1" scene_dir="${ROOT}/scene_$(printf '%04d' "$1")"
  batch_tool decision "${scene_dir}" "${REQUEST_TEMPLATE}" "${seed}"
}

RENDER_REQUESTED=1
[[ "${RENDER}" == "1" ]] || RENDER_REQUESTED=0
REVIEW_REQUESTED=1
[[ "${REVIEW}" == "1" ]] || REVIEW_REQUESTED=0
export BATCH_RENDER="${RENDER_REQUESTED}" BATCH_REVIEW="${REVIEW_REQUESTED}"
export BATCH_REVIEW_REPAIR="${REVIEW_REPAIR}" BATCH_REVIEW_STRICT="${REVIEW_STRICT}"

if [[ "${SUMMARY_ONLY}" != "1" ]]; then
  # the request every scene of this batch is bound to: fingerprints the code,
  # the asset registry and the config once, at the time the batch starts
  WORK_DIR="$(mktemp -d)"
  REQUEST_TEMPLATE="${WORK_DIR}/request.json"
  batch_tool template "${REQUEST_TEMPLATE}" \
    || { echo "batch_generate: cannot fingerprint the request" >&2; exit 1; }

  for seed in "${SEEDS[@]}"; do
    scene_dir="${ROOT}/scene_$(printf '%04d' "${seed}")"
    mkdir -p "${scene_dir}"
    echo
    if [[ "${RESUME}" == "1" ]]; then
      decision="$(resume_decision "${seed}")" \
        || decision=$'rerun\tmanifest tool failed (see the error above)'
      verdict="${decision%%$'\t'*}"
      detail="${decision#*$'\t'}"
      if [[ "${verdict}" == "resume" ]]; then
        RESUMED_SEEDS+=("${seed}")
        echo "--- seed ${seed} -> ${scene_dir} --- (resumed: ${detail})"
        continue
      fi
      echo "--- seed ${seed} -> ${scene_dir} --- (rerun: ${detail})"
    else
      echo "--- seed ${seed} -> ${scene_dir} ---"
    fi
    ARGS=("${seed}" "${scene_dir}")
    [[ -n "${ROOMS}" ]] && ARGS+=("${ROOMS}")
    status=0
    set +e
    if [[ "${RENDER}" == "1" ]]; then
      PANOWORLD_RUN_ID="${BATCH_ID}-seed${seed}" \
        bash "${SCRIPT_DIR}/layout_gen_run.sh" "${ARGS[@]}" > "${scene_dir}/run.log" 2>&1
    else
      PANOWORLD_RUN_ID="${BATCH_ID}-seed${seed}" PANOWORLD_SKIP_RENDER=1 \
        bash "${SCRIPT_DIR}/layout_gen_run.sh" "${ARGS[@]}" > "${scene_dir}/run.log" 2>&1
    fi
    status=$?
    set -e
    if [[ ${status} -ne 0 ]]; then
      tail -n 3 "${scene_dir}/run.log" || true
      echo "batch_generate: seed ${seed} failed (exit ${status}), continuing" >&2
    else
      tail -n 1 "${scene_dir}/run.log" || true
    fi
    # the manifest describes what this run produced, whatever its grade: it is
    # the record the next resume compares its request and spec hash against
    batch_tool manifest "${scene_dir}" "${REQUEST_TEMPLATE}" "${seed}" \
      "${BATCH_ID}" "${BATCH_ID}-seed${seed}" \
      || echo "batch_generate: seed ${seed}: could not write ${scene_dir}/resume_manifest.json" >&2
  done
fi

RESUMED_CSV=""
if (( ${#RESUMED_SEEDS[@]} > 0 )); then
  RESUMED_CSV="$(IFS=,; echo "${RESUMED_SEEDS[*]}")"
fi

set +e
"${HOSTPY}" - "${ROOT}" "${START}" "${RENDER}" "${BATCH_ID}" "${STARTED_AT}" \
  "${RESUME}" "${SUMMARY_ONLY}" "${RESUMED_CSV}" "${REVIEW}" "${REQUEST_TEMPLATE:-}" \
  "${SEEDS[@]}" <<'PY'
import datetime, json, os, sys

root, start, render, batch_id, started_at = sys.argv[1:6]
resume, summary_only = sys.argv[6] == "1", sys.argv[7] == "1"
resumed = {int(value) for value in sys.argv[8].split(",") if value}
review_requested = sys.argv[9] == "1"
request_template_path = sys.argv[10]
seeds = [int(value) for value in sys.argv[11:]]
render_requested = render == "1"

request_template = None
if request_template_path and os.path.isfile(request_template_path):
    with open(request_template_path, encoding="utf-8") as handle:
        request_template = json.load(handle)

levels = {"layout_ok": 0, "render_ok": 0, "validation_ok": 0, "bundle_ok": 0}
if not render_requested:
    levels["render_ok"] = None
    levels["validation_ok"] = None
if review_requested:
    levels["review_ok"] = 0
failure_reasons = {}
failed_checks = {}
review_issues = {}
rows = []
degraded_seeds = []


def render_quality(summary):
    """The profile a scene was asked for against the one that produced its
    artifacts: a retry that succeeded at a reduced profile is a *degraded*
    bundle, never a full-quality completion."""
    render = summary.get("render") or {}
    requested = {"resolution": render.get("resolution"), "samples": render.get("samples")}
    effective = None
    if render.get("status") in ("ok", "validation_flagged"):
        attempts = [attempt for attempt in (render.get("attempts") or []) if attempt]
        if attempts:
            last = attempts[-1]
            effective = {"resolution": last.get("resolution"), "samples": last.get("samples"),
                         "attempt": last.get("attempt"), "status": last.get("status")}
    degraded = bool(effective and requested["resolution"]
                    and (effective["resolution"] != requested["resolution"]
                         or (effective["samples"] is not None
                             and requested["samples"] is not None
                             and effective["samples"] != requested["samples"])))
    return {"requested": requested, "effective": effective, "degraded": degraded}


def load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def family_block(scene_dir, summary):
    """What the scene asked for and built: drawn family, fallback, attempts.

    ``layout_gen.py`` records the arrangement on its own stream
    (``stats.auto_pick``), whether the sticky draw had to give up
    (``stats.auto_fallback``) and how many attempts the seed needed.  The batch
    summary carries them per row so a reader (and
    ``tools/diversity_report.py success``) can tell the *requested* family mix
    from the delivered one without opening every layout report."""
    report = load_json(os.path.join(scene_dir, "layout_report.json"))
    report = report if isinstance(report, dict) else {}
    stats = report.get("stats") if isinstance(report.get("stats"), dict) else None
    if not stats:
        stats = summary.get("stats") if isinstance(summary.get("stats"), dict) else {}
    auto_pick = stats.get("auto_pick")
    if not isinstance(auto_pick, (list, tuple)):
        auto_pick = None
    attempts = stats.get("attempts_used")
    if attempts is None and stats.get("attempt") is not None:
        attempts = int(stats["attempt"]) + 1
    return {
        "built": stats.get("family") or report.get("family") or report.get("shape")
        or stats.get("shape"),
        "requested_shape": stats.get("shape") or report.get("shape"),
        "auto_pick": [auto_pick[0], auto_pick[1]] if auto_pick else None,
        "auto_fallback": None if stats.get("auto_fallback") is None
        else bool(stats.get("auto_fallback")),
        "attempts_used": attempts,
        "floors": stats.get("floors"),
    }


def bundle_problems(scene_dir, render_expected, review_expected):
    """Final artifacts a bundle_ok row must still have on disk: the stage
    reports plus, per camera of the spec, a non-empty viewpoint directory."""
    needed = ["spec.json", "checks_report.json"]
    if render_expected:
        needed.append("validation_report.json")
    if review_expected:
        needed.append("review_report.json")
    missing = [name for name in needed if not os.path.isfile(os.path.join(scene_dir, name))]
    if render_expected:
        viewpoints = os.path.join(scene_dir, "scene", "viewpoints")
        if not os.path.isdir(viewpoints) or not os.listdir(viewpoints):
            missing.append("scene/viewpoints")
            return missing
        cameras = []
        spec_path = os.path.join(scene_dir, "spec.json")
        try:
            with open(spec_path, encoding="utf-8") as handle:
                cameras = [str(camera.get("name"))
                           for camera in (json.load(handle).get("cameras") or [])]
        except Exception:
            cameras = []
        for name in cameras:
            view = os.path.join(viewpoints, name)
            if not os.path.isdir(view):
                missing.append("scene/viewpoints/%s" % name)
            elif not os.listdir(view):
                missing.append("scene/viewpoints/%s (empty)" % name)
    return missing


def legacy_ok(summary, scene_dir, render_expected):
    """Grade a pre-schema-2 run_summary.json (no "ok" block)."""
    checks_pass = bool((summary.get("checks") or {}).get("pass"))
    layout_ok = checks_pass and os.path.isfile(os.path.join(scene_dir, "spec.json"))
    status = (summary.get("render") or {}).get("status")
    validation = summary.get("validation")
    validation_ok = None if validation is None else bool(
        validation.get("pass_at_resolution", validation.get("pass")))
    if not render_expected or status in (None, "skipped_by_request"):
        render_ok = None
        bundle_ok = layout_ok
    elif status in ("skipped_low_memory", "killed_signal_9"):
        render_ok = False
        bundle_ok = False
    elif status in ("ok", "validation_flagged"):
        render_ok = True
        bundle_ok = bool(layout_ok and validation_ok)
    else:
        render_ok = False
        bundle_ok = False
    return {"layout_ok": bool(layout_ok), "render_ok": render_ok,
            "validation_ok": validation_ok, "bundle_ok": bool(bundle_ok)}


for seed in seeds:
    scene_dir = os.path.join(root, "scene_%04d" % seed)
    expected_run_id = "%s-seed%d" % (batch_id, seed)
    row = {
        "seed": seed,
        "dir": os.path.relpath(scene_dir, root),
        "status": "failed",
        "resumed": seed in resumed,
        "run_id": None,
        "render_requested": render_requested,
        "stages": {"layout": "failed", "checks": "not_run",
                   "render": "not_run", "validation": "not_run"},
        "ok": {"layout_ok": False, "render_ok": None, "validation_ok": None, "bundle_ok": False},
        "failure": None,
    }
    summary_path = os.path.join(scene_dir, "run_summary.json")
    summary = None
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
        except Exception as error:
            summary = None
            row["failure"] = {"stage": "summary", "kind": "unreadable_summary", "detail": str(error)}
    if summary is None:
        if row["failure"] is None:
            row["failure"] = {"stage": "layout", "kind": "missing_summary",
                              "detail": "no run_summary.json (layout_gen or the interpreter failed)"}
    else:
        # summary-only mode re-reads scenes that earlier batches produced, so
        # the current batch id cannot be expected there; a normal run always
        # holds a summary written by the child it just launched
        fresh = summary_only or seed in resumed or summary.get("run_id") == expected_run_id
        legacy = "ok" not in summary
        if not fresh and not legacy:
            row["failure"] = {"stage": "summary", "kind": "stale_summary",
                              "detail": "run_summary.json run_id %r does not match this batch (%s)"
                                        % (summary.get("run_id"), expected_run_id)}
        else:
            row["run_id"] = summary.get("run_id")
            if legacy:
                row["legacy_summary"] = True
                ok = legacy_ok(summary, scene_dir, render_requested)
                status = (summary.get("render") or {}).get("status")
                stages = {
                    "layout": "ok" if ok["layout_ok"] else "failed",
                    "checks": "ok" if (summary.get("checks") or {}).get("pass") else "failed",
                    "render": status or "unknown",
                    "validation": "ok" if ok["validation_ok"] else
                                  ("not_run" if ok["validation_ok"] is None else "failed"),
                }
                failure = None if ok["bundle_ok"] else {
                    "stage": "legacy", "kind": "legacy_summary_not_bundle_ok",
                    "detail": "pre-schema-2 summary without a bundle_ok record"}
            else:
                ok = summary.get("ok") or {}
                stages = summary.get("stages") or row["stages"]
                failure = summary.get("failure") if not ok.get("bundle_ok") else None
            row["stages"] = stages
            row["ok"] = ok
            row["failure"] = failure
            if ok.get("bundle_ok"):
                problems = bundle_problems(scene_dir, render_requested, review_requested)
                if problems:
                    row["ok"] = dict(ok, bundle_ok=False)
                    row["failure"] = {"stage": "bundle", "kind": "bundle_artifacts_missing",
                                      "detail": "missing " + ", ".join(problems)}

    # legacy per-scene fields, kept for readers of the old summary format
    if summary is not None:
        row["rooms"] = summary.get("rooms")
        row["room_types"] = summary.get("room_types")
        row["cameras"] = len(summary.get("cameras") or [])
        row["furniture"] = summary.get("furniture")
        row["generation_attempts"] = (summary.get("stats") or {}).get("attempts_used")
        row["family"] = family_block(scene_dir, summary)
        row["checks_pass"] = bool((summary.get("checks") or {}).get("pass"))
        row["render"] = (summary.get("render") or {}).get("status")
        row["render_quality"] = render_quality(summary)
        if row["render_quality"]["degraded"]:
            row["degraded"] = True
            degraded_seeds.append(seed)
        validation = summary.get("validation")
        row["validation_pass"] = None if validation is None else bool(validation.get("pass"))
        row["validation_pass_at_resolution"] = None if validation is None \
            else bool(validation.get("pass_at_resolution", validation.get("pass")))
        if review_requested:
            review = summary.get("review") or {}
            row["review_pass"] = row["ok"].get("review_ok")
            row["review_score"] = review.get("score")
            row["review_issues"] = review.get("issues") or {}
            row["review_failed_checks"] = review.get("failed_checks") or []
            row["review_repair"] = (review.get("repair") or {}).get("status")

    ok = row["ok"]
    if ok.get("bundle_ok"):
        row["status"] = "bundle_ok"
    elif ok.get("layout_ok") and row["stages"].get("render") == "pending_retry":
        row["status"] = "pending_retry"
    else:
        row["status"] = "failed"

    if ok.get("layout_ok"):
        levels["layout_ok"] += 1
    if render_requested:
        if ok.get("render_ok") is True:
            levels["render_ok"] += 1
        if ok.get("validation_ok") is True:
            levels["validation_ok"] += 1
    if review_requested and ok.get("review_ok") is True:
        levels["review_ok"] += 1
    if ok.get("bundle_ok"):
        levels["bundle_ok"] += 1

    failure = row.get("failure")
    if row["status"] != "bundle_ok" and failure:
        key = "%s:%s" % (failure.get("stage", "unknown"), failure.get("kind", "unknown"))
        failure_reasons[key] = failure_reasons.get(key, 0) + 1
        for name in failure.get("checks", []) or []:
            failed_checks[name] = failed_checks.get(name, 0) + 1
    if review_requested and row["status"] != "bundle_ok":
        for name in row.get("review_failed_checks") or []:
            review_issues[name] = review_issues.get(name, 0) + 1
    rows.append(row)

pending_retry = sum(1 for row in rows if row["status"] == "pending_retry")
failed = len(rows) - levels["bundle_ok"] - pending_retry

# The family mix of the batch: the arrangements it really built, the fallbacks
# behind them, and how many rows predate the contract fields.  tools/
# diversity_report.py success compares this with the drawn families (per row) and
# with the delivered products, instead of counting the request as the result.
family_mix = {}
family_fallback = []
family_unknown = 0
for row in rows:
    block = row.get("family") or {}
    name = block.get("built")
    if name:
        family_mix[name] = family_mix.get(name, 0) + 1
    else:
        family_unknown += 1
    if block.get("auto_fallback") is True:
        family_fallback.append(row["seed"])

summary = {
    "kind": "layout_gen_batch",
    "schema_version": 2,
    "batch_id": batch_id,
    "started_at": started_at,
    "finished_at": None,
    "start_seed": start,
    "count": len(seeds),
    "seeds": seeds,
    "render_requested": render_requested,
    "review_requested": review_requested,
    "resume": resume,
    "summary_only": summary_only,
    "request": request_template,
    "manifest": {
        "file": "resume_manifest.json",
        "schema_version": 1,
        "note": "written per scene by batch_generate.sh: the request, the code/asset "
                "fingerprints and the spec hash a PANOWORLD_BATCH_RESUME=1 run compares "
                "against before it keeps an existing bundle",
        "verified_resumed": len(resumed),
    },
    "bundle_definition": "layout_ok when the render was not requested; "
                         "layout_ok + render_ok + validation_ok when it was"
                         + ("; plus review_ok when the review was requested" if review_requested else ""),
    "levels": levels,
    "passed": levels["bundle_ok"],
    "degraded": len(degraded_seeds),
    "degraded_seeds": degraded_seeds,
    "degraded_definition": "bundle_ok scenes whose render succeeded at a reduced profile "
                           "after a memory retry (see each row's render_quality): they did not "
                           "deliver the requested resolution/samples",
    "pending_retry": pending_retry,
    "failed": failed,
    "failure_reasons": failure_reasons,
    "failed_checks": failed_checks,
    "review_issues": review_issues,
    "families": {
        "built": dict(sorted(family_mix.items())),
        "unknown": family_unknown,
        "requested_shape": ((request_template or {}).get("layout") or {}).get("shape"),
        "auto_fallback": len(family_fallback),
        "auto_fallback_seeds": family_fallback,
        "definition": "families.built counts each row's family.built (the arrangement the "
                      "scene was really built as, not the one it asked for); rows without a "
                      "family block predate the contract.  See each row's family for the "
                      "drawn family (auto_pick), the auto fallback and the attempts",
    },
    "scenes": rows,
}
summary["finished_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
with open(os.path.join(root, "batch_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print()
print("== batch summary ==")
for row in rows:
    ok = row["ok"]
    render_note = ""
    if ok.get("validation_ok") and row.get("validation_pass_at_resolution") is not None \
            and row.get("validation_pass") is False and row.get("validation_pass_at_resolution"):
        render_note = " [canonical_resolution]"
    quality = row.get("render_quality") or {}
    if quality.get("degraded"):
        render_note += " [degraded %s, %s samples at the requested %s, %s samples]" % (
            (quality["effective"] or {}).get("resolution"),
            (quality["effective"] or {}).get("samples"),
            (quality["requested"] or {}).get("resolution"),
            (quality["requested"] or {}).get("samples"))
    review_note = ""
    if review_requested:
        if row.get("review_pass"):
            issues = row.get("review_issues") or {}
            review_note = " review=PASS(%sw)" % (issues.get("warn") or 0)
        elif row.get("review_pass") is False:
            review_note = " review=FAIL(%s)" % ",".join(row.get("review_failed_checks") or [])[:40]
        else:
            review_note = " review=not_run"
    detail = ""
    if row["status"] != "bundle_ok" and row.get("failure"):
        detail = "  <- %s: %s" % (("%s:%s" % (row['failure'].get('stage'), row['failure'].get('kind'))),
                                  row["failure"].get("detail", ""))
    print("  seed %-4d %-13s%s layout=%s render=%s validation=%s%s family=%s rooms=%s cams=%s "
          "furniture=%s attempts=%s%s%s"
          % (row["seed"], row["status"], " (resumed)" if row["resumed"] else "",
             row["stages"].get("layout"), row["stages"].get("render"),
             row["stages"].get("validation"), review_note,
             (row.get("family") or {}).get("built") or "-",
             len(row.get("rooms") or []), row.get("cameras"), row.get("furniture"),
             row.get("generation_attempts"), render_note, detail))
if render_requested:
    review_level = ", review_ok %d/%d" % (levels.get("review_ok"), len(rows)) if review_requested else ""
    print("  levels: layout_ok %d/%d, render_ok %d/%d, validation_ok %d/%d%s, bundle_ok %d/%d "
          "(pending_retry %d, failed %d)"
          % (levels["layout_ok"], len(rows), levels["render_ok"], len(rows),
             levels["validation_ok"], len(rows), review_level, levels["bundle_ok"], len(rows),
             pending_retry, failed))
    if degraded_seeds:
        print("  degraded: %d/%d bundle_ok scenes rendered at a reduced profile (seeds %s)"
              % (len(degraded_seeds), len(rows),
                 ",".join(str(seed) for seed in degraded_seeds[:10])
                 + (",..." if len(degraded_seeds) > 10 else "")))
else:
    review_level = ", review_ok %d/%d" % (levels.get("review_ok"), len(rows)) if review_requested else ""
    print("  levels: layout_ok %d/%d%s, bundle_ok %d/%d (render not requested; pending_retry %d, failed %d)"
          % (levels["layout_ok"], len(rows), review_level, levels["bundle_ok"], len(rows),
             pending_retry, failed))
if family_mix:
    fallback_note = " (auto fallback %d)" % len(family_fallback) if family_fallback else ""
    print("  families(built): %s%s"
          % (", ".join("%s %d" % (name, family_mix[name]) for name in sorted(family_mix)),
             fallback_note))
if family_unknown:
    print("  families(built): %d row(s) carry no family block (pre-contract summary)" % family_unknown)
for key in sorted(failure_reasons):
    print("  failure %-40s %d" % (key, failure_reasons[key]))
if review_requested:
    for key in sorted(review_issues):
        print("  review issue %-33s %d" % (key, review_issues[key]))
print("  passed(bundle_ok) %d/%d" % (levels["bundle_ok"], len(rows)))
raise SystemExit(0 if levels["bundle_ok"] == len(rows) else 1)
PY
BATCH_STATUS=$?
set -e

echo
if [[ ${BATCH_STATUS} -eq 0 ]]; then
  echo "batch_generate finished: ${ROOT}/batch_summary.json (all scenes bundle_ok)"
else
  echo "batch_generate finished: ${ROOT}/batch_summary.json (some scenes are not bundle_ok)" >&2
fi
exit "${BATCH_STATUS}"
