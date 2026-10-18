#!/usr/bin/env bash
# Install the Habitat-Sim navigation environment used by roomgen_nav.
#
# The environment lives in third_party/nav_env (gitignored).  It is only for
# NavMesh/pathfinding and the discrete PointNav runtime; RGB-D observations are
# rendered by Blender, so no X server or EGL device is required at runtime.
#
# Usage: bash scripts/setup_nav_env.sh [--force]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${REPO}/third_party/nav_env"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

if [[ -x "${ENV_DIR}/bin/python" && "${FORCE}" == "0" ]]; then
  echo "nav env already present: ${ENV_DIR}"
else
  command -v conda >/dev/null || { echo "conda is required" >&2; exit 1; }
  conda create -y -p "${ENV_DIR}" -c aihabitat -c conda-forge \
    "habitat-sim=0.3.3" headless libegl python=3.9 numpy
fi

"${ENV_DIR}/bin/python" - <<'PY'
import habitat_sim
print("habitat_sim", habitat_sim.__version__)
PY
echo "nav env ready: ${ENV_DIR}"
echo "run with:"
echo "  LD_LIBRARY_PATH=\"${ENV_DIR}/lib:\${LD_LIBRARY_PATH:-}\" ${ENV_DIR}/bin/python -m roomgen_nav.eval_pointnav --help"
