#!/usr/bin/env bash
# layout_gen: one seed -> scene spec -> placement checks -> (memory permitting)
# the full Blender pipeline -> per-scene summary.
#
# 用法: bash scripts/layout_gen_run.sh <seed> <outdir> [rooms]
#   rooms  optional 2..5; by default the seed decides
#
# Writes into <outdir>: spec.json, layout_report.json, checks_report.json,
# run_summary.json and -- when the render runs -- build/, scene/,
# validation_report.json.
#
# Environment:
#   PANOWORLD_RES / PANOWORLD_SAMPLES   render settings (default 1024x512, 32)
#   PANOWORLD_SKIP_RENDER=1             layout + checks only
#   PANOWORLD_MIN_FREE_MB               free memory required before rendering
#                                       (default 2200 MB: a 2048x1024 Cycles
#                                       process peaks near 1.9 GB on this host)
#   PANOWORLD_WAIT_FOR_MEMORY=1         wait for that headroom instead of
#                                       skipping, up to PANOWORLD_MEM_TIMEOUT_S
#   PANOWORLD_HOSTPY, PANOWORLD_BLENDER as in blender_agent_run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOSTPY="${PANOWORLD_HOSTPY:-/opt/conda/envs/diffaero/bin/python}"
SEED="${1:?usage: layout_gen_run.sh <seed> <outdir> [rooms]}"
OUT="${2:?usage: layout_gen_run.sh <seed> <outdir> [rooms]}"
ROOMS="${3:-}"
SKIP_RENDER="${PANOWORLD_SKIP_RENDER:-0}"
CONFIG="${PANOWORLD_LAYOUT_CONFIG:-${REPO}/configs/layout_gen_demo.json}"

# render defaults come from configs/layout_gen_demo.json unless the environment
# overrides them; the helper prints an empty line when the key or file is absent
config_value() {
  [[ -r "${CONFIG}" ]] || return 0
  "${HOSTPY}" -c 'import json,sys
try:
    node = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print(""); raise SystemExit
for key in sys.argv[2].split("."):
    node = node.get(key, {}) if isinstance(node, dict) else {}
print(node if not isinstance(node, dict) else "")' "${CONFIG}" "$1" 2>/dev/null || true
}

RES="${PANOWORLD_RES:-$(config_value render.resolution)}"
SAMPLES="${PANOWORLD_SAMPLES:-$(config_value render.samples)}"
MIN_FREE_MB="${PANOWORLD_MIN_FREE_MB:-$(config_value render.min_free_mb)}"
RES="${RES:-1024x512}"
SAMPLES="${SAMPLES:-32}"
MIN_FREE_MB="${MIN_FREE_MB:-2200}"

mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
SPEC="${OUT}/spec.json"
REPORT="${OUT}/layout_report.json"
CHECKS="${OUT}/checks_report.json"

# Headroom of this container's cgroup, in MB.
#
# A second agent renders in the same cgroup, so this decides whether starting a
# Cycles process is safe.  The cgroup counts page cache as usage; the kernel
# reclaims that before it OOM kills anything, so what matters is the anonymous
# RSS (memory.stat "rss" in cgroup v1) against the limit.  Falls back to
# memory.current / memory.max (v2) and then to the host's MemAvailable.
mem_available_mb() {
  local limit="" rss="" usage=""
  for candidate in /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory.max; do
    [[ -r "${candidate}" ]] && limit="$(cat "${candidate}" 2>/dev/null || true)"
  done
  if [[ -r /sys/fs/cgroup/memory/memory.stat ]]; then
    rss="$(awk '/^rss /{print $2}' /sys/fs/cgroup/memory/memory.stat 2>/dev/null || true)"
  fi
  if [[ -n "${limit}" && "${limit}" != "max" && -n "${rss}" ]] && (( limit > rss )); then
    echo $(( (limit - rss) / 1048576 ))
    return
  fi
  [[ -r /sys/fs/cgroup/memory.current ]] && usage="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || true)"
  if [[ -n "${limit}" && "${limit}" != "max" && -n "${usage}" ]] && (( limit > usage )); then
    echo $(( (limit - usage) / 1048576 ))
    return
  fi
  free -m | awk '/^MemAvailable:/{print $2} /^Mem:/{print $7}'
}

echo "== layout_gen =="
echo "   seed    : ${SEED}"
echo "   out     : ${OUT}"
echo "   rooms   : ${ROOMS:-<seed>}"
echo "   render  : ${RES}, ${SAMPLES} samples (skip=${SKIP_RENDER})"

echo
echo "--- [1/3] generate the layout ---"
GEN_ARGS=(--seed "${SEED}" --out "${SPEC}" --report "${REPORT}")
[[ -n "${ROOMS}" ]] && GEN_ARGS+=(--rooms "${ROOMS}")
"${HOSTPY}" "${REPO}/blender_agent/layout_gen.py" "${GEN_ARGS[@]}"

echo
echo "--- [2/3] placement checks ---"
set +e
"${HOSTPY}" "${REPO}/blender_agent/checks.py" "${SPEC}" --report "${CHECKS}"
CHECKS_STATUS=$?
set -e

RENDER_STATUS="skipped"
PIPELINE_STATUS=0
VALIDATION_JSON="null"
if [[ "${SKIP_RENDER}" != "1" && ${CHECKS_STATUS} -eq 0 ]]; then
  FREE_MB="$(mem_available_mb)"
  echo
  echo "--- [3/3] blender pipeline (free memory: ${FREE_MB} MB, need ${MIN_FREE_MB} MB) ---"
  if (( FREE_MB < MIN_FREE_MB )) && [[ "${PANOWORLD_WAIT_FOR_MEMORY:-0}" == "1" ]]; then
    deadline=$(( SECONDS + ${PANOWORLD_MEM_TIMEOUT_S:-1800} ))
    while (( FREE_MB < MIN_FREE_MB && SECONDS < deadline )); do
      sleep 20
      FREE_MB="$(mem_available_mb)"
      echo "layout_gen_run.sh: waiting for memory (have ${FREE_MB} MB, need ${MIN_FREE_MB} MB)"
    done
  fi
  if (( FREE_MB < MIN_FREE_MB )); then
    RENDER_STATUS="skipped_low_memory"
    echo "layout_gen_run.sh: only ${FREE_MB} MB free (< ${MIN_FREE_MB} MB), skipping the render."
    echo "                  re-run later or set PANOWORLD_MIN_FREE_MB to force it."
  else
    # The validator exits non-zero whenever a check fails -- and at any
    # resolution other than the canonical 2048x1024 its own
    # ``canonical_resolution`` check fails by construction.  That is a property
    # of the render profile, not of the layout, so it must not abort the run:
    # the report below reads validation_report.json and applies the resolution
    # rule (only ``canonical_resolution`` may fail below 2048x1024).
    set +e
    PANOWORLD_RES="${RES}" PANOWORLD_SAMPLES="${SAMPLES}" \
      bash "${REPO}/scripts/blender_agent_run.sh" "${SPEC}" "${OUT}"
    PIPELINE_STATUS=$?
    set -e
    if [[ ${PIPELINE_STATUS} -eq 0 ]]; then
      RENDER_STATUS="ok"
    elif [[ ${PIPELINE_STATUS} -ge 128 ]]; then
      RENDER_STATUS="killed_signal_$((PIPELINE_STATUS - 128))"
      echo "layout_gen_run.sh: the blender pipeline died on signal $((PIPELINE_STATUS - 128))" >&2
    else
      RENDER_STATUS="validation_flagged"
    fi
  fi
  if [[ -f "${OUT}/validation_report.json" ]]; then
    VALIDATION_JSON="\"${OUT}/validation_report.json\""
  fi
else
  [[ "${SKIP_RENDER}" == "1" ]] && RENDER_STATUS="skipped_by_request"
  [[ ${CHECKS_STATUS} -ne 0 ]] && RENDER_STATUS="skipped_checks_failed"
fi

"${HOSTPY}" - "${OUT}" "${SEED}" "${RENDER_STATUS}" "${CHECKS_STATUS}" "${RES}" "${SAMPLES}" \
  "${PIPELINE_STATUS}" <<'PY'
import json, os, sys

out, seed, render_status, checks_status, res, samples, pipeline_status = sys.argv[1:8]
summary = {"kind": "layout_gen_run", "seed": int(seed), "out": out,
           "render": {"status": render_status, "resolution": res, "samples": int(samples),
                      "pipeline_exit_code": int(pipeline_status)}}
spec_path = os.path.join(out, "spec.json")
if os.path.isfile(spec_path):
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    summary["rooms"] = [room["id"] for room in spec["rooms"]]
    summary["room_types"] = [room["type"] for room in spec["rooms"]]
    summary["cameras"] = [camera["name"] for camera in spec["cameras"]]
    summary["furniture"] = len(spec.get("furniture", []))
    summary["openings"] = len(spec.get("openings", []))
    summary["windows"] = len(spec.get("windows", []))
report_path = os.path.join(out, "layout_report.json")
if os.path.isfile(report_path):
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    summary["stats"] = report.get("stats", {})
    summary["opening_0"] = report.get("opening_0")
    summary["door_visual_forecast"] = report.get("door_visual_forecast", {})
    summary["checks"] = {"pass": (report.get("checks") or {}).get("pass")}
if int(checks_status) != 0:
    summary["checks"] = {"pass": False}
validation_path = os.path.join(out, "validation_report.json")
if os.path.isfile(validation_path):
    with open(validation_path, encoding="utf-8") as handle:
        validation = json.load(handle)
    failed = [check["name"] for check in validation.get("checks", []) if not check["pass"]]
    canonical = res == "2048x1024"
    summary["validation"] = {
        "pass": validation.get("pass"),
        "pass_at_resolution": bool(validation.get("pass"))
        or (failed == ["canonical_resolution"] and not canonical),
        "canonical_resolution": canonical,
        "failed": failed,
        "depth": [{"camera": entry["camera"], "median_mm": round(entry["median_error_m"] * 1000, 3),
                   "p95_mm": round(entry["p95_error_m"] * 1000, 3),
                   "valid_fraction": round(entry["valid_fraction"], 5),
                   "door_ratio": round(entry["door_spot_check"].get("ratio", 0.0), 3)}
                  for entry in next((check for check in validation.get("checks", [])
                                     if check["name"] == "depth_matches_geometry"), {}).get("cameras", [])],
    }
with open(os.path.join(out, "run_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
validation = summary.get("validation", {})
print(f"layout_gen: seed {seed}: checks={'PASS' if summary.get('checks', {}).get('pass') else 'FAIL'} "
      f"render={render_status} "
      f"validation={validation.get('pass')} "
      f"(at {res}: {'PASS' if validation.get('pass_at_resolution', True) else 'FAIL'}"
      + (f", failed {validation.get('failed')}" if validation.get("failed") else "") + ")")
PY

echo
echo "layout_gen finished: ${OUT}"
[[ ${CHECKS_STATUS} -eq 0 ]] || exit "${CHECKS_STATUS}"
