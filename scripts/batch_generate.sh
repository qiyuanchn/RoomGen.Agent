#!/usr/bin/env bash
# Batch layout generation: one directory per seed plus an aggregate table.
#
# 用法: bash scripts/batch_generate.sh <start_seed> <count> <out_root>
#
# Layout: <out_root>/scene_<seed>/{spec.json, layout_report.json,
#                                   checks_report.json, run_summary.json,
#                                   validation_report.json, run.log, ...}
#         <out_root>/batch_summary.json
#
# batch_summary.json (schema_version 2) grades every scene on four levels --
# layout_ok / render_ok / validation_ok / bundle_ok -- instead of folding
# layout-only runs and memory-skipped renders into one pass count.  "passed"
# counts bundle_ok only: a render that was requested but did not complete is
# pending_retry (or failed), never a pass; failure_reasons / failed_checks
# carry the distribution.  A failing scene never stops the batch, but the
# script exits 0 only when every scene reached bundle_ok (1 otherwise).
#
# Environment:
#   PANOWORLD_BATCH_RENDER=1        also run the Blender pipeline for every
#                                   scene (default 0: layout + checks only,
#                                   so a 20 seed sweep is seconds long)
#   PANOWORLD_BATCH_RESUME=1        keep scenes whose run_summary.json is a
#                                   complete bundle of the same mode instead
#                                   of re-running them (interrupted batch)
#   PANOWORLD_BATCH_SUMMARY_ONLY=1  rebuild batch_summary.json from the
#                                   existing scene dirs and run nothing
#   PANOWORLD_BATCH_SEEDS=0,7,13    explicit seed list (START/COUNT ignored)
#   PANOWORLD_RES / PANOWORLD_SAMPLES / PANOWORLD_MIN_FREE_MB / PANOWORLD_ROOMS
#                                   forwarded to layout_gen_run.sh
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
BATCH_ID="${PANOWORLD_BATCH_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "${ROOT}"
ROOT="$(cd "${ROOT}" && pwd)"
echo "== batch_generate =="
echo "   batch id: ${BATCH_ID}"
echo "   seeds   : ${START} .. $(( START + COUNT - 1 ))"
echo "   out     : ${ROOT}"
echo "   render  : ${RENDER} (resume=${RESUME}, summary_only=${SUMMARY_ONLY})"

SEEDS=()
if [[ -n "${SEED_LIST}" ]]; then
  IFS=',' read -r -a SEEDS <<< "${SEED_LIST}"
else
  for (( offset = 0; offset < COUNT; offset++ )); do
    SEEDS+=("$(( START + offset ))")
  done
fi

RESUMED_SEEDS=()

# Prints "resume" when an existing scene holds a complete bundle of the
# requested mode (and the artifacts are still on disk), "rerun" otherwise.
resume_decision() { # <seed> <render_requested>
  "${HOSTPY}" - "${ROOT}" "$1" "$2" <<'PY'
import json, os, sys

root, seed, render_expected = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
scene_dir = os.path.join(root, "scene_%04d" % seed)
summary_path = os.path.join(scene_dir, "run_summary.json")
if not os.path.isfile(summary_path):
    print("rerun"); raise SystemExit
try:
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
except Exception:
    print("rerun"); raise SystemExit
ok = summary.get("ok") or {}
if not ok.get("bundle_ok") or bool(summary.get("render_requested")) != render_expected:
    print("rerun"); raise SystemExit
needed = ["spec.json", "checks_report.json"]
if render_expected:
    needed.append("validation_report.json")
missing = [name for name in needed if not os.path.isfile(os.path.join(scene_dir, name))]
if render_expected:
    viewpoints = os.path.join(scene_dir, "scene", "viewpoints")
    if not os.path.isdir(viewpoints) or not os.listdir(viewpoints):
        missing.append("scene/viewpoints")
print("rerun" if missing else "resume")
PY
}

RENDER_REQUESTED=1
[[ "${RENDER}" == "1" ]] || RENDER_REQUESTED=0

if [[ "${SUMMARY_ONLY}" != "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    scene_dir="${ROOT}/scene_$(printf '%04d' "${seed}")"
    mkdir -p "${scene_dir}"
    echo
    if [[ "${RESUME}" == "1" ]] && [[ "$(resume_decision "${seed}" "${RENDER_REQUESTED}")" == "resume" ]]; then
      RESUMED_SEEDS+=("${seed}")
      echo "--- seed ${seed} -> ${scene_dir} --- (resumed: existing bundle kept)"
      continue
    fi
    echo "--- seed ${seed} -> ${scene_dir} ---"
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
  done
fi

RESUMED_CSV=""
if (( ${#RESUMED_SEEDS[@]} > 0 )); then
  RESUMED_CSV="$(IFS=,; echo "${RESUMED_SEEDS[*]}")"
fi

set +e
"${HOSTPY}" - "${ROOT}" "${START}" "${RENDER}" "${BATCH_ID}" "${STARTED_AT}" \
  "${RESUME}" "${SUMMARY_ONLY}" "${RESUMED_CSV}" "${SEEDS[@]}" <<'PY'
import datetime, json, os, sys

root, start, render, batch_id, started_at = sys.argv[1:6]
resume, summary_only = sys.argv[6] == "1", sys.argv[7] == "1"
resumed = {int(value) for value in sys.argv[8].split(",") if value}
seeds = [int(value) for value in sys.argv[9:]]
render_requested = render == "1"

levels = {"layout_ok": 0, "render_ok": 0, "validation_ok": 0, "bundle_ok": 0}
if not render_requested:
    levels["render_ok"] = None
    levels["validation_ok"] = None
failure_reasons = {}
failed_checks = {}
rows = []


def bundle_problems(scene_dir, render_expected):
    """Final artifacts a bundle_ok row must still have on disk."""
    needed = ["spec.json", "checks_report.json"]
    if render_expected:
        needed.append("validation_report.json")
    missing = [name for name in needed if not os.path.isfile(os.path.join(scene_dir, name))]
    if render_expected:
        viewpoints = os.path.join(scene_dir, "scene", "viewpoints")
        if not os.path.isdir(viewpoints) or not os.listdir(viewpoints):
            missing.append("scene/viewpoints")
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
                problems = bundle_problems(scene_dir, render_requested)
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
        row["checks_pass"] = bool((summary.get("checks") or {}).get("pass"))
        row["render"] = (summary.get("render") or {}).get("status")
        validation = summary.get("validation")
        row["validation_pass"] = None if validation is None else bool(validation.get("pass"))
        row["validation_pass_at_resolution"] = None if validation is None \
            else bool(validation.get("pass_at_resolution", validation.get("pass")))

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
    if ok.get("bundle_ok"):
        levels["bundle_ok"] += 1

    failure = row.get("failure")
    if row["status"] != "bundle_ok" and failure:
        key = "%s:%s" % (failure.get("stage", "unknown"), failure.get("kind", "unknown"))
        failure_reasons[key] = failure_reasons.get(key, 0) + 1
        for name in failure.get("checks", []) or []:
            failed_checks[name] = failed_checks.get(name, 0) + 1
    rows.append(row)

pending_retry = sum(1 for row in rows if row["status"] == "pending_retry")
failed = len(rows) - levels["bundle_ok"] - pending_retry
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
    "resume": resume,
    "summary_only": summary_only,
    "bundle_definition": "layout_ok when the render was not requested; "
                         "layout_ok + render_ok + validation_ok when it was",
    "levels": levels,
    "passed": levels["bundle_ok"],
    "pending_retry": pending_retry,
    "failed": failed,
    "failure_reasons": failure_reasons,
    "failed_checks": failed_checks,
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
    detail = ""
    if row["status"] != "bundle_ok" and row.get("failure"):
        detail = "  <- %s: %s" % (("%s:%s" % (row['failure'].get('stage'), row['failure'].get('kind'))),
                                  row["failure"].get("detail", ""))
    print("  seed %-4d %-13s%s layout=%s render=%s validation=%s rooms=%s cams=%s furniture=%s attempts=%s%s%s"
          % (row["seed"], row["status"], " (resumed)" if row["resumed"] else "",
             row["stages"].get("layout"), row["stages"].get("render"),
             row["stages"].get("validation"),
             len(row.get("rooms") or []), row.get("cameras"), row.get("furniture"),
             row.get("generation_attempts"), render_note, detail))
if render_requested:
    print("  levels: layout_ok %d/%d, render_ok %d/%d, validation_ok %d/%d, bundle_ok %d/%d "
          "(pending_retry %d, failed %d)"
          % (levels["layout_ok"], len(rows), levels["render_ok"], len(rows),
             levels["validation_ok"], len(rows), levels["bundle_ok"], len(rows),
             pending_retry, failed))
else:
    print("  levels: layout_ok %d/%d, bundle_ok %d/%d (render not requested; pending_retry %d, failed %d)"
          % (levels["layout_ok"], len(rows), levels["bundle_ok"], len(rows), pending_retry, failed))
for key in sorted(failure_reasons):
    print("  failure %-40s %d" % (key, failure_reasons[key]))
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
