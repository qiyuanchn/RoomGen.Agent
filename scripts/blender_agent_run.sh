#!/usr/bin/env bash
# blender_agent: build a PanoWorld scene in Blender, export viewpoints, validate.
#
# 用法: bash scripts/blender_agent_run.sh [spec] [outdir]
#   spec   default configs/blender_agent_p0_demo.json
#          P1 furnished home: configs/blender_agent_p1_furnished.json
#   outdir default outputs/blender_agent/demo_house
#
# Environment overrides:
#   PANOWORLD_BLENDER, PANOWORLD_HOSTPY   tool paths
#   PANOWORLD_SAMPLES                     Cycles samples (default 128)
#   PANOWORLD_RES                         WxH, e.g. 1024x512 for a fast look (default 2048x1024)
#   PANOWORLD_CACHE                       Poly Haven asset cache (default third_party/assets/polyhaven)
#   PANOWORLD_TEXRES                      furniture texture variant (1k/2k/4k)
#   PANOWORLD_EXPORT_GLB                  1 = also write build/scene.glb (browser 3D viewer)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

BLENDER="${PANOWORLD_BLENDER:-${REPO}/third_party/blender-4.5.14-linux-x64/blender}"
HOSTPY="${PANOWORLD_HOSTPY:-/opt/conda/envs/diffaero/bin/python}"
SPEC="${1:-${REPO}/configs/blender_agent_p0_demo.json}"
OUT="${2:-${REPO}/outputs/blender_agent/demo_house}"
SAMPLES="${PANOWORLD_SAMPLES:-128}"
RES="${PANOWORLD_RES:-2048x1024}"
WIDTH="${RES%x*}"
HEIGHT="${RES#*x}"
CACHE="${PANOWORLD_CACHE:-${REPO}/third_party/assets/polyhaven}"
TEXRES="${PANOWORLD_TEXRES:-1k}"
# PANOWORLD_EXPORT_GLB=1 also writes scene.glb for the browser 3D viewer
# (scripts/scene3d_view.sh); it adds ~1 minute and a few dozen MB per scene.
GLB_ARGS=()
if [[ "${PANOWORLD_EXPORT_GLB:-0}" == "1" ]]; then
  GLB_ARGS=(--export-glb)
fi

for tool in "${BLENDER}" "${HOSTPY}"; do
  if [[ ! -x "${tool}" ]]; then
    echo "blender_agent_run.sh: not executable: ${tool}" >&2
    exit 1
  fi
done

mkdir -p "${OUT}"
echo "== blender_agent =="
echo "   spec    : ${SPEC}"
echo "   out     : ${OUT}"
echo "   render  : ${WIDTH}x${HEIGHT}, ${SAMPLES} samples (AgX, OIDN post-pass)"
echo "   blender : ${BLENDER}"
echo "   host py : ${HOSTPY}"
echo "   cache   : ${CACHE}"
echo "   glb     : ${PANOWORLD_EXPORT_GLB:-0} (1 = write build/scene.glb for the 3D viewer)"

echo
echo "--- [1/6] ERP orientation self-test ---"
"${BLENDER}" --background --factory-startup \
  --python "${REPO}/blender_agent/render_selftest.py" \
  -- --out "${OUT}/build" --width "${WIDTH}" --height "${HEIGHT}"

echo
echo "--- [2/6] asset libraries (Poly Haven, separate Blender process) ---"
# Building a library resets Blender's file and leaves ~250 MB behind, memory the
# render process cannot afford on this host, so it happens here instead.
"${BLENDER}" --background --factory-startup \
  --python "${REPO}/blender_agent/run_build.py" \
  -- --spec "${SPEC}" --out "${OUT}/build" --cache "${CACHE}" \
     --tex-resolution "${TEXRES}" --prepare-assets-only 2>&1 | tail -3 || true

echo
echo "--- [3/6] render (Blender, Cycles CPU) ---"
# Rendered without in-render denoising on purpose: OIDN inside the 2048x1024
# render process exceeds this container's memory cap and gets OOM-killed, so
# the linear EXR is denoised in step 3 by a separate, much smaller process.
# (Use run_build.py without --no-denoise on hosts with more headroom.)
set +e
"${BLENDER}" --background --factory-startup \
  --python "${REPO}/blender_agent/run_build.py" \
  -- --spec "${SPEC}" --out "${OUT}/build" --no-denoise --cache "${CACHE}" \
     --width "${WIDTH}" --height "${HEIGHT}" --samples "${SAMPLES}" "${GLB_ARGS[@]}"
STATUS=$?
set -e
if [[ ${STATUS} -ne 0 ]]; then
  echo "blender_agent_run.sh: build exited with ${STATUS}, retrying once" >&2
  "${BLENDER}" --background --factory-startup \
    --python "${REPO}/blender_agent/run_build.py" \
    -- --spec "${SPEC}" --out "${OUT}/build" --no-denoise --cache "${CACHE}" \
       --width "${WIDTH}" --height "${HEIGHT}" --samples "${SAMPLES}" "${GLB_ARGS[@]}"
fi

echo
echo "--- [4/6] OpenImageDenoise pass (separate Blender process) ---"
"${BLENDER}" --background --factory-startup \
  --python "${REPO}/blender_agent/run_build.py" \
  -- --spec "${SPEC}" --out "${OUT}/build" --denoise-only

echo
echo "--- [5/6] export PanoWorld viewpoints ---"
"${HOSTPY}" "${REPO}/blender_agent/export_scene.py" \
  --build "${OUT}/build" --spec "${SPEC}" --out "${OUT}/scene" --force

echo
echo "--- [6/6] numeric validation ---"
"${HOSTPY}" "${REPO}/tools/validate_blender_agent_p0.py" \
  --build "${OUT}/build" --scene "${OUT}/scene" --spec "${SPEC}" \
  --report "${OUT}/validation_report.json"

echo
echo "blender_agent P0 finished: ${OUT}"
