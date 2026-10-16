#!/usr/bin/env bash
# Fetch the RoomGen.Agent asset library into third_party/assets/ and rebuild
# the registries the layout generator and builder read.
#
# Usage:
#   bash scripts/fetch_assets.sh                 # Poly Haven (CC0, ~200 MB) + MIDI-3D (~13.3 GB)
#   bash scripts/fetch_assets.sh --skip-midi3d   # Poly Haven only (fast)
#
# Licensing: Poly Haven is CC0; the MIDI-3D extract (processed 3D-FRONT /
# 3D-FUTURE) is CC-BY-NC-4.0 — research/evaluation only, NOT commercial.
# See README for swapping in the official 3D-FRONT + 3D-FUTURE set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSETS_DIR="${REPO_ROOT}/third_party/assets"
export ASSETS_DIR
HOSTPY="${PANOWORLD_HOSTPY:-python3}"
SKIP_MIDI3D=0
[[ "${1:-}" == "--skip-midi3d" ]] && SKIP_MIDI3D=1

mkdir -p "${ASSETS_DIR}"

echo "== Poly Haven (CC0) =="
mapfile -t PH_REFS < <("${HOSTPY}" - <<'PY'
import json, os, sys
path = os.path.join(os.environ["ASSETS_DIR"], "merged_index.json")
if not os.path.isfile(path):
    sys.exit(0)  # first run: nothing merged yet, fetch by demo-spec needs
reg = json.load(open(path))
models = reg.get("models", reg)
for name, entry in sorted(models.items()):
    if entry.get("source") == "polyhaven":
        print(entry.get("ref") or f"ph:{name}")
PY
)
if [[ ${#PH_REFS[@]} -gt 0 ]]; then
  "${HOSTPY}" "${REPO_ROOT}/blender_agent/assets.py" fetch --resolution 1k \
    --cache "${ASSETS_DIR}/polyhaven" --assets "${PH_REFS[@]}"
else
  # First run: fetch the assets the shipped demo specs reference.
  "${HOSTPY}" "${REPO_ROOT}/blender_agent/assets.py" fetch --resolution 1k \
    --cache "${ASSETS_DIR}/polyhaven" \
    --assets ph:oak_wood_planks ph:beige_wall_001 ph:poly_wool_herringbone \
             ph:sofa_02 ph:modern_coffee_table_01 ph:modern_arm_chair_01 \
             ph:GothicBed_01 ph:old_bed_frame ph:painted_wooden_nightstand \
             ph:wooden_bookshelf_worn ph:book_encyclopedia_set_01 \
             ph:modern_ceiling_lamp_01 ph:desk_lamp_arm_01 \
             ph:potted_plant_01 ph:potted_plant_02 ph:wicker_basket_01 \
             ph:hanging_picture_frame_01
fi

if [[ "${SKIP_MIDI3D}" == "1" ]]; then
  echo "== MIDI-3D skipped (--skip-midi3d) =="
else
  echo "== MIDI-3D TEST subset (CC-BY-NC-4.0! ~13.3 GB) =="
  cd "${ASSETS_DIR}"
  if [[ ! -f 3D-FRONT-TEST-SCENE.tar.gz ]]; then
    curl -L --http1.1 --retry 5 --retry-all-errors -C - \
      -o 3D-FRONT-TEST-SCENE.tar.gz \
      "https://huggingface.co/datasets/huanngzh/3D-Front/resolve/main/3D-FRONT-TEST-SCENE.tar.gz"
  fi
  if [[ ! -d 3D-FRONT-TEST-SCENE ]]; then
    tar xzf 3D-FRONT-TEST-SCENE.tar.gz
  fi
  mkdir -p midi3d
  "${HOSTPY}" "${REPO_ROOT}/tools/index_midi3d.py" \
    --root 3D-FRONT-TEST-SCENE --out midi3d/index_raw.json --curate-out midi3d/index_curated.json
fi

echo "== rebuild registries =="
"${HOSTPY}" "${REPO_ROOT}/tools/make_merged_assets.py" \
  --polyhaven "${ASSETS_DIR}/polyhaven/index.json" \
  --midi3d "${ASSETS_DIR}/midi3d/index_curated.json" \
  --out "${ASSETS_DIR}/merged_index.json" \
  --midi3d-cache-index "${ASSETS_DIR}/midi3d/index.json"

echo "assets ready under ${ASSETS_DIR}"
