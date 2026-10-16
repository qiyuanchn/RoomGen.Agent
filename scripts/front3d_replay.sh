#!/usr/bin/env bash
# blender_agent P3: replay an official 3D-FRONT scene as a PanoWorld viewpoint set.
#
#   bash scripts/front3d_replay.sh <scene.json> <3d-future-model-dir> <outdir> \
#        [3d-front-textures-dir]
#
#   scene.json             a 3D-FRONT scene JSON (3D-FRONT/<uid>.json)
#   3d-future-model-dir    3D-FUTURE-model (one folder per jid: raw_model.obj,
#                          texture.png, model.mtl)
#   outdir                 build + scene + reports land here
#   3d-front-textures-dir  optional 3D-FRONT-textures (wall/floor textures); when
#                          omitted the dataset's flat colours are used
#
# Environment overrides:
#   PANOWORLD_BLENDER, PANOWORLD_HOSTPY   tool paths
#   PANOWORLD_SAMPLES                     Cycles samples (default 32)
#   PANOWORLD_RES                         WxH (default 2048x1024; 1024x512 is the
#                                         fast iteration profile)
#   PANOWORLD_CAMERAS                     room | snapshots (default room: one pano
#                                         per room at eye height)
#   PANOWORLD_EYE_HEIGHT                  camera height for room cameras (1.5)
#   PANOWORLD_SKIP_VALIDATE              1 = skip the numeric validation stages
#
# What it does (same pipeline as scripts/blender_agent_run.sh, fed by the dataset):
#   orientation self-test -> build (dataset shell + 3D-FUTURE models + sky/camera)
#   -> OpenImageDenoise pass -> export PanoWorld viewpoints -> validation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 3 ]]; then
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi

SCENE_JSON="$1"
MODEL_ROOT="$2"
OUT="$3"
TEXTURE_ROOT="${4:-}"

BLENDER="${PANOWORLD_BLENDER:-${REPO}/third_party/blender-4.5.14-linux-x64/blender}"
HOSTPY="${PANOWORLD_HOSTPY:-/opt/conda/envs/diffaero/bin/python}"
SAMPLES="${PANOWORLD_SAMPLES:-32}"
RES="${PANOWORLD_RES:-2048x1024}"
CAMERAS="${PANOWORLD_CAMERAS:-room}"
EYE_HEIGHT="${PANOWORLD_EYE_HEIGHT:-1.5}"
WIDTH="${RES%x*}"
HEIGHT="${RES#*x}"

for tool in "${BLENDER}" "${HOSTPY}"; do
  if [[ ! -x "${tool}" ]]; then
    echo "front3d_replay.sh: not executable: ${tool}" >&2
    exit 1
  fi
done
if [[ ! -f "${SCENE_JSON}" ]]; then
  echo "front3d_replay.sh: no such scene JSON: ${SCENE_JSON}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "front3d_replay.sh: no such model root: ${MODEL_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUT}"
SPEC="${OUT}/front3d_scene_spec.json"

echo "== 3D-FRONT replay =="
echo "   scene   : ${SCENE_JSON}"
echo "   models  : ${MODEL_ROOT}"
echo "   textures: ${TEXTURE_ROOT:-<none: dataset colours>}"
echo "   out     : ${OUT}"
echo "   render  : ${WIDTH}x${HEIGHT}, ${SAMPLES} samples, cameras=${CAMERAS}"

echo
echo "--- [1/7] parse the scene and bridge it to a scene spec ---"
"${HOSTPY}" "${REPO}/blender_agent/front3d.py" \
  --scene "${SCENE_JSON}" --model-root "${MODEL_ROOT}" \
  ${TEXTURE_ROOT:+--texture-root "${TEXTURE_ROOT}"} \
  --camera-source "${CAMERAS}" --eye-height "${EYE_HEIGHT}" \
  --spec-out "${SPEC}" --dry-run

echo
echo "--- [2/7] ERP orientation self-test ---"
"${BLENDER}" --background --factory-startup \
  --python "${REPO}/blender_agent/render_selftest.py" \
  -- --out "${OUT}/build" --width "${WIDTH}" --height "${HEIGHT}"

echo
echo "--- [3/7] build + render (dataset shell, 3D-FUTURE models) ---"
# rendered without in-render denoising: OIDN in the render process exceeds this
# container's memory cap, so the linear EXR is denoised by step 4 in its own,
# much smaller process
set +e
"${BLENDER}" --background --factory-startup \
  --python "${REPO}/blender_agent/run_build.py" \
  -- --front3d-scene "${SCENE_JSON}" \
     --front3d-model-root "${MODEL_ROOT}" \
     ${TEXTURE_ROOT:+--front3d-texture-root "${TEXTURE_ROOT}"} \
     --front3d-camera-source "${CAMERAS}" \
     --front3d-eye-height "${EYE_HEIGHT}" \
     --out "${OUT}/build" --no-denoise \
     --width "${WIDTH}" --height "${HEIGHT}" --samples "${SAMPLES}"
STATUS=$?
set -e
if [[ ${STATUS} -ne 0 ]]; then
  echo "front3d_replay.sh: build exited with ${STATUS}, retrying once" >&2
  "${BLENDER}" --background --factory-startup \
    --python "${REPO}/blender_agent/run_build.py" \
    -- --front3d-scene "${SCENE_JSON}" \
       --front3d-model-root "${MODEL_ROOT}" \
       ${TEXTURE_ROOT:+--front3d-texture-root "${TEXTURE_ROOT}"} \
       --front3d-camera-source "${CAMERAS}" \
       --front3d-eye-height "${EYE_HEIGHT}" \
       --out "${OUT}/build" --no-denoise \
       --width "${WIDTH}" --height "${HEIGHT}" --samples "${SAMPLES}"
fi

echo
echo "--- [4/7] OpenImageDenoise pass (separate Blender process) ---"
"${BLENDER}" --background --factory-startup \
  --python "${REPO}/blender_agent/run_build.py" \
  -- --out "${OUT}/build" --denoise-only

# the build records the spec it used; the host tools must address that one
SPEC="${OUT}/build/front3d_scene_spec.json"

echo
echo "--- [5/7] export PanoWorld viewpoints ---"
"${HOSTPY}" "${REPO}/blender_agent/export_scene.py" \
  --build "${OUT}/build" --spec "${SPEC}" --out "${OUT}/scene" --force

if [[ "${PANOWORLD_SKIP_VALIDATE:-0}" == "1" ]]; then
  echo
  echo "front3d_replay.sh: validation skipped (PANOWORLD_SKIP_VALIDATE=1)"
  echo "3D-FRONT replay finished: ${OUT}"
  exit 0
fi

echo
echo "--- [6/7] numeric validation (poses, depth vs geometry, map) ---"
set +e
"${HOSTPY}" "${REPO}/tools/validate_blender_agent_p0.py" \
  --build "${OUT}/build" --scene "${OUT}/scene" --spec "${SPEC}" \
  --report "${OUT}/validation_report.json"
set -e
# below the canonical 2048x1024 the validator's own canonical_resolution check is
# the only one allowed to fail; at the canonical size everything must pass
"${HOSTPY}" "${REPO}/tests/eval_validation_report.py" \
  --report "${OUT}/validation_report.json" --resolution "${RES}"

echo
echo "--- [7/7] front3d checks (shell, models, poses, metric depth) ---"
"${HOSTPY}" "${REPO}/tests/check_front3d_build.py" \
  --build "${OUT}/build" --scene "${OUT}/scene" --spec "${SPEC}" \
  --report "${OUT}/front3d_check_report.json"

echo
echo "3D-FRONT replay finished: ${OUT}"
