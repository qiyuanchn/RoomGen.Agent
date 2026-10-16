#!/usr/bin/env bash
# scene3d_view: serve the browser 3D viewer (three.js + scene.glb) for one scene.
#
# 用法: bash scripts/scene3d_view.sh <scene_dir> [--port 8004] [--host 0.0.0.0]
#   scene_dir  a scene/build directory that contains scene.glb, e.g.
#              outputs/blender_agent/p1_settled        (GLB inside build/)
#              outputs/gen_assets                       (GLB inside build/)
#
# The page (tools/scene3d_viewer/) loads scene.glb over three.js and offers
# Orbit + Fly/Walk modes plus jumps to the panorama camera poses. The panorama
# viewer in PanoWorld serves its own ERP tour on 8003; this one is 3D.
#
# Produce the GLB first (a few minutes, geometry only):
#   bash scripts/blender_agent_run.sh <spec.json> <outdir>   # + --export-glb, or
#   third_party/blender-4.5.14-linux-x64/blender --background --factory-startup \
#     --python blender_agent/run_build.py -- \
#     --spec <spec.json> --out <outdir>/build --skip-render --export-glb
#
# Environment overrides:
#   PANOWORLD_HOSTPY  python for the server (default python3; stdlib only)
#   SCENE3D_HOST      bind address (default 0.0.0.0)
#   SCENE3D_PORT      bind port (default 8004)
# HOST / PORT are intentionally *not* inherited: this host's HOST variable is
# often a broken container hostname, which would break the bind.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOSTPY="${PANOWORLD_HOSTPY:-python3}"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  [[ $# -ge 1 ]] && exit 0 || exit 2
fi

SCENE="$1"
shift

HOST="${SCENE3D_HOST:-0.0.0.0}"
PORT="${SCENE3D_PORT:-8004}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    --host) HOST="${2:?--host needs a value}"; shift 2 ;;
    --host=*) HOST="${1#*=}"; shift ;;
    *) echo "scene3d_view.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! command -v "${HOSTPY}" >/dev/null 2>&1; then
  echo "scene3d_view.sh: python not found: ${HOSTPY}" >&2
  exit 1
fi

echo "== scene3d_view =="
echo "   scene : ${SCENE}"
echo "   bind  : ${HOST}:${PORT}"
echo "   page  : ${REPO}/tools/scene3d_viewer/"
echo

exec "${HOSTPY}" "${REPO}/tools/scene3d_viewer/serve.py" "${SCENE}" --host "${HOST}" --port "${PORT}"
