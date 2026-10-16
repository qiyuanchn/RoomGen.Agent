#!/usr/bin/env bash
# Batch layout generation: one directory per seed plus an aggregate table.
#
# 用法: bash scripts/batch_generate.sh <start_seed> <count> <out_root>
#
# Layout: <out_root>/scene_<seed>/{spec.json, layout_report.json,
#                                   checks_report.json, run_summary.json,
#                                   validation_report.json, run.log}
#         <out_root>/batch_summary.json
#
# A failing scene never stops the batch: the summary records the seed, the
# stage that failed ("spec", "checks", "render", "validation") and the checks
# violation text, and the loop moves on.
#
# Environment:
#   PANOWORLD_BATCH_RENDER=1   also run the Blender pipeline for every scene
#                              (default 0: layout + checks only, so a 20 seed
#                              sweep is seconds instead of hours)
#   PANOWORLD_RES / PANOWORLD_SAMPLES / PANOWORLD_MIN_FREE_MB / PANOWORLD_ROOMS
#                              forwarded to layout_gen_run.sh
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

mkdir -p "${ROOT}"
ROOT="$(cd "${ROOT}" && pwd)"
echo "== batch_generate =="
echo "   seeds   : ${START} .. $(( START + COUNT - 1 ))"
echo "   out     : ${ROOT}"
echo "   render  : ${RENDER}"

SEEDS=()
if [[ -n "${SEED_LIST}" ]]; then
  IFS=',' read -r -a SEEDS <<< "${SEED_LIST}"
else
  for (( offset = 0; offset < COUNT; offset++ )); do
    SEEDS+=("$(( START + offset ))")
  done
fi

for seed in "${SEEDS[@]}"; do
  scene_dir="${ROOT}/scene_$(printf '%04d' "${seed}")"
  mkdir -p "${scene_dir}"
  echo
  echo "--- seed ${seed} -> ${scene_dir} ---"
  ARGS=("${seed}" "${scene_dir}")
  [[ -n "${ROOMS}" ]] && ARGS+=("${ROOMS}")
  set +e
  if [[ "${RENDER}" == "1" ]]; then
    bash "${SCRIPT_DIR}/layout_gen_run.sh" "${ARGS[@]}" > "${scene_dir}/run.log" 2>&1
  else
    PANOWORLD_SKIP_RENDER=1 bash "${SCRIPT_DIR}/layout_gen_run.sh" "${ARGS[@]}" \
      > "${scene_dir}/run.log" 2>&1
  fi
  status=$?
  set -e
  tail -n 2 "${scene_dir}/run.log" || true
  if [[ ${status} -ne 0 ]]; then
    echo "batch_generate: seed ${seed} failed (stage status ${status}), continuing" >&2
  fi
done

"${HOSTPY}" - "${ROOT}" "${START}" "${RENDER}" "${SEEDS[@]}" <<'PY'
import json, os, sys

root, start, render, seeds = sys.argv[1], int(sys.argv[2]), sys.argv[3], [int(v) for v in sys.argv[4:]]
rows = []
for seed in seeds:
    scene_dir = os.path.join(root, "scene_%04d" % seed)
    row = {"seed": seed, "dir": os.path.relpath(scene_dir, root), "status": "pass", "stage": None}
    summary_path = os.path.join(scene_dir, "run_summary.json")
    if not os.path.isfile(summary_path):
        row.update(status="fail", stage="layout",
                   detail="no run_summary.json (layout_gen or the interpreter failed)")
        rows.append(row)
        continue
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    row["rooms"] = summary.get("rooms")
    row["room_types"] = summary.get("room_types")
    row["cameras"] = len(summary.get("cameras") or [])
    row["furniture"] = summary.get("furniture")
    row["generation_attempts"] = (summary.get("stats") or {}).get("attempts_used")
    row["checks_pass"] = bool((summary.get("checks") or {}).get("pass"))
    row["render"] = (summary.get("render") or {}).get("status")
    validation = summary.get("validation")
    row["validation_pass"] = None if validation is None else bool(validation.get("pass"))
    row["validation_pass_at_resolution"] = None if validation is None \
        else bool(validation.get("pass_at_resolution", validation.get("pass")))
    if not row["checks_pass"]:
        row.update(status="fail", stage="checks",
                   detail=" ".join(summary.get("checks", {}).get("violations", [])[:3]))
        checks_path = os.path.join(scene_dir, "checks_report.json")
        if os.path.isfile(checks_path):
            with open(checks_path, encoding="utf-8") as handle:
                row["detail"] = " | ".join(json.load(handle).get("violations", [])[:3]) or row["detail"]
    elif row["render"] in (None, "skipped_by_request", "skipped_low_memory"):
        row["status"] = "pass_layout_only"
    elif row["validation_pass_at_resolution"] is False:
        row.update(status="fail", stage="validation", detail=", ".join(validation.get("failed", [])))
    elif row["validation_pass"] is None:
        row.update(status="fail", stage="render", detail="render produced no validation_report.json")
    rows.append(row)

passed = sum(1 for row in rows if row["status"] in ("pass", "pass_layout_only"))
summary = {
    "kind": "layout_gen_batch",
    "start_seed": start,
    "count": len(seeds),
    "seeds": seeds,
    "render_requested": render == "1",
    "passed": passed,
    "failed": len(rows) - passed,
    "scenes": rows,
}
with open(os.path.join(root, "batch_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print()
print("== batch summary ==")
for row in rows:
    detail = f"  <- {row['stage']}: {row.get('detail')}" if row["stage"] else ""
    render_note = ""
    if row.get("validation_pass") and row.get("validation_pass_at_resolution") is False:
        render_note = " [canonical_resolution]"
    print("  seed %-4d %-18s rooms=%s cams=%s furniture=%s attempts=%s%s%s"
          % (row["seed"], row["status"], len(row.get("rooms") or []), row.get("cameras"),
             row.get("furniture"), row.get("generation_attempts"), render_note, detail))
print("  passed %d/%d" % (passed, len(rows)))
PY

echo
echo "batch_generate finished: ${ROOT}/batch_summary.json"
