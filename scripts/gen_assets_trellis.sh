#!/usr/bin/env bash
# gen_assets_trellis: fill the "gen:" asset cache, then build a scene from it.
#
# 用法: bash scripts/gen_assets_trellis.sh [outdir] [backend]
#   outdir  default outputs/gen_assets
#   backend default $PANOWORLD_GEN_BACKEND, else "auto":
#             trellis when a GPU container with the TRELLIS checkout is found,
#             proc (procedural trimesh furniture) otherwise
#
# Stages:
#   1. tools/gen_assets_trellis.py generate  -> third_party/assets/generated
#      (glbs + index.json, i.e. the gen:<id> assets the spec references)
#   2. tools/gen_assets_trellis.py emit-spec -> <outdir>/scene_spec.json
#      (the shipped P1 shell + the generated furniture, plus the provenance
#       <outdir>/scene_spec.assets.json and the placement check report)
#   3. scripts/blender_agent_run.sh          -> build, render, export, validate
#
# Environment overrides:
#   PANOWORLD_GEN_BACKEND   auto | proc | trellis          (default auto)
#   PANOWORLD_GEN_CLASSES   classes to fill               (default sofa bed chair table cabinet)
#   PANOWORLD_GEN_VARIANTS  assets per class              (default: 2 trellis / 1 proc)
#   PANOWORLD_GEN_FORCE     1 = regenerate cached assets
#   PANOWORLD_TRELLIS_CONTAINER / _REPO / _PYTHON / _HF   GPU container + TRELLIS paths
#   PANOWORLD_RES, PANOWORLD_SAMPLES, PANOWORLD_BLENDER, PANOWORLD_HOSTPY
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOSTPY="${PANOWORLD_HOSTPY:-/opt/conda/envs/diffaero/bin/python}"
OUT="${1:-${REPO}/outputs/gen_assets}"
BACKEND="${2:-${PANOWORLD_GEN_BACKEND:-auto}}"
CLASSES="${PANOWORLD_GEN_CLASSES:-sofa bed chair table cabinet}"
RES="${PANOWORLD_RES:-1024x512}"
SAMPLES="${PANOWORLD_SAMPLES:-32}"

GEN_ARGS=(generate --backend "${BACKEND}" --classes ${CLASSES})
if [[ -n "${PANOWORLD_GEN_VARIANTS:-}" ]]; then
  GEN_ARGS+=(--variants "${PANOWORLD_GEN_VARIANTS}")
fi
if [[ "${PANOWORLD_GEN_FORCE:-0}" == "1" ]]; then
  GEN_ARGS+=(--force)
fi
if [[ -n "${PANOWORLD_TRELLIS_CONTAINER:-}" ]]; then
  GEN_ARGS+=(--container "${PANOWORLD_TRELLIS_CONTAINER}")
fi

mkdir -p "${OUT}"
echo "== gen_assets_trellis =="
echo "   out     : ${OUT}"
echo "   backend : ${BACKEND}"
echo "   classes : ${CLASSES}"
echo "   render  : ${RES}, ${SAMPLES} samples"
echo "   host py : ${HOSTPY}"

echo
echo "--- [1/3] generate gen: assets ---"
"${HOSTPY}" "${REPO}/tools/gen_assets_trellis.py" "${GEN_ARGS[@]}"

echo
echo "--- [2/3] emit the demo spec + provenance ---"
"${HOSTPY}" "${REPO}/tools/gen_assets_trellis.py" emit-spec --out "${OUT}/scene_spec.json"

echo
echo "--- [3/3] build, render, export, validate (standard pipeline) ---"
PANOWORLD_RES="${RES}" PANOWORLD_SAMPLES="${SAMPLES}" \
  bash "${REPO}/scripts/blender_agent_run.sh" "${OUT}/scene_spec.json" "${OUT}"

echo
echo "gen_assets_trellis finished: ${OUT}"
echo "  spec        : ${OUT}/scene_spec.json"
echo "  provenance  : ${OUT}/scene_spec.assets.json   (which backend built each gen: glb)"
echo "  placement   : ${OUT}/checks_report.json"
echo "  validation  : ${OUT}/validation_report.json"
