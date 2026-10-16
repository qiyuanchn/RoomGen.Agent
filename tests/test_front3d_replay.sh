#!/usr/bin/env bash
# blender_agent P3: 3D-FRONT replay acceptance test.
#
#   bash tests/test_front3d_replay.sh [fixture_dir] [outdir]
#
# fixturedir  default tests/fixtures/front3d_mini (hand-built two-room scene)
# outdir      default outputs/blender_agent/tests/front3d_mini
#
# Stages, in order:
#   1  unit tests (quaternion / transform order / Y-up->Z-up / known poses)
#   2  host parse + spec bridge (front3d.py --dry-run --spec-out)
#   3  ERP orientation self-test
#   4  Blender build: dataset shell (inline mesh triangles) + 3D-FUTURE models
#   5  OpenImageDenoise pass (separate Blender process)
#   6  host export to the PanoWorld viewpoint layout
#   7  numeric validation (tools/validate_blender_agent_p0.py)
#   8  front3d-specific numeric checks (tests/check_front3d_build.py)
#
# Environment overrides: PANOWORLD_BLENDER, PANOWORLD_HOSTPY,
# PANOWORLD_RES (default 1024x512 -- the low-resolution iteration profile),
# PANOWORLD_SAMPLES (default 32).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

BLENDER="${PANOWORLD_BLENDER:-${REPO}/third_party/blender-4.5.14-linux-x64/blender}"
HOSTPY="${PANOWORLD_HOSTPY:-/opt/conda/envs/diffaero/bin/python}"
FIXTURE="${1:-${REPO}/tests/fixtures/front3d_mini}"
OUT="${2:-${REPO}/outputs/blender_agent/tests/front3d_mini}"
SCENE_JSON="${FIXTURE}/scene.json"
MODEL_ROOT="${FIXTURE}/3D-FUTURE-model"
TEXTURE_ROOT="${FIXTURE}/3D-FRONT-textures"
SAMPLES="${PANOWORLD_SAMPLES:-32}"
RES="${PANOWORLD_RES:-1024x512}"
WIDTH="${RES%x*}"
HEIGHT="${RES#*x}"
CANONICAL="2048x1024"

for tool in "${BLENDER}" "${HOSTPY}"; do
  if [[ ! -x "${tool}" ]]; then
    echo "test_front3d_replay.sh: not executable: ${tool}" >&2
    exit 1
  fi
done
if [[ ! -f "${SCENE_JSON}" ]]; then
  echo "test_front3d_replay.sh: no scene JSON at ${SCENE_JSON}" >&2
  exit 1
fi

mkdir -p "${OUT}"
SPEC="${OUT}/front3d_scene_spec.json"

echo "== 3D-FRONT replay test =="
echo "   scene   : ${SCENE_JSON}"
echo "   models  : ${MODEL_ROOT}"
echo "   out     : ${OUT}"
echo "   render  : ${WIDTH}x${HEIGHT}, ${SAMPLES} samples"

echo
echo "--- [1/8] unit tests (host python) ---"
"${HOSTPY}" "${REPO}/tests/test_front3d_math.py"

echo
echo "--- [2/8] parse + spec bridge ---"
"${HOSTPY}" "${REPO}/blender_agent/front3d.py" \
  --scene "${SCENE_JSON}" --model-root "${MODEL_ROOT}" --texture-root "${TEXTURE_ROOT}" \
  --dry-run --spec-out "${SPEC}"

echo
echo "--- [3/8] ERP orientation self-test ---"
"${BLENDER}" --background --factory-startup \
  --python "${REPO}/blender_agent/render_selftest.py" \
  -- --out "${OUT}/build" --width "${WIDTH}" --height "${HEIGHT}"

echo
echo "--- [4/8] build + render (dataset shell, 3D-FUTURE models) ---"
# no in-render denoising on purpose: OIDN inside the render process exceeds this
# container's memory cap (see blender_agent/README.md)
set +e
"${BLENDER}" --background --factory-startup \
  --python "${REPO}/blender_agent/run_build.py" \
  -- --front3d-scene "${SCENE_JSON}" \
     --front3d-model-root "${MODEL_ROOT}" \
     --front3d-texture-root "${TEXTURE_ROOT}" \
     --out "${OUT}/build" --no-denoise \
     --width "${WIDTH}" --height "${HEIGHT}" --samples "${SAMPLES}"
STATUS=$?
set -e
if [[ ${STATUS} -ne 0 ]]; then
  echo "test_front3d_replay.sh: build exited with ${STATUS}, retrying once" >&2
  "${BLENDER}" --background --factory-startup \
    --python "${REPO}/blender_agent/run_build.py" \
    -- --front3d-scene "${SCENE_JSON}" \
       --front3d-model-root "${MODEL_ROOT}" \
       --front3d-texture-root "${TEXTURE_ROOT}" \
       --out "${OUT}/build" --no-denoise \
       --width "${WIDTH}" --height "${HEIGHT}" --samples "${SAMPLES}"
fi

echo
echo "--- [5/8] OpenImageDenoise pass ---"
"${BLENDER}" --background --factory-startup \
  --python "${REPO}/blender_agent/run_build.py" \
  -- --out "${OUT}/build" --denoise-only

# run_build writes the spec it actually built into the build directory; the two
# entry paths (--front3d-scene here, --spec for a bridged spec) must agree, and
# the host tools address the one the build recorded.
BUILD_SPEC="${OUT}/build/front3d_scene_spec.json"
if ! diff -q "${SPEC}" "${BUILD_SPEC}" >/dev/null; then
  echo "test_front3d_replay.sh: the bridged spec differs from the one the build used" >&2
  diff "${SPEC}" "${BUILD_SPEC}" | head -20 >&2
  exit 1
fi
echo "bridged spec identical to the build's own: ${BUILD_SPEC}"
SPEC="${BUILD_SPEC}"

echo
echo "--- [6/8] export PanoWorld viewpoints ---"
"${HOSTPY}" "${REPO}/blender_agent/export_scene.py" \
  --build "${OUT}/build" --spec "${SPEC}" --out "${OUT}/scene" --force

echo
echo "--- [7/8] numeric validation (depth vs geometry, poses, map) ---"
set +e
"${HOSTPY}" "${REPO}/tools/validate_blender_agent_p0.py" \
  --build "${OUT}/build" --scene "${OUT}/scene" --spec "${SPEC}" \
  --report "${OUT}/validation_report.json"
set -e

# the iteration profile renders below the canonical 2048x1024, so the validator's
# own canonical_resolution check is the only one allowed to fail there
"${HOSTPY}" "${REPO}/tests/eval_validation_report.py" \
  --report "${OUT}/validation_report.json" --resolution "${RES}"

echo
echo "--- [8/8] front3d numeric checks (shell, models, poses, metric depth) ---"
"${HOSTPY}" "${REPO}/tests/check_front3d_build.py" \
  --build "${OUT}/build" --scene "${OUT}/scene" --spec "${SPEC}" \
  --report "${OUT}/front3d_check_report.json"

echo
echo "3D-FRONT replay test PASS: ${OUT}"
