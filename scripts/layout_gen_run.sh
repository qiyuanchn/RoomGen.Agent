#!/usr/bin/env bash
# layout_gen: one seed -> scene spec -> placement checks -> (memory permitting)
# the full Blender pipeline -> per-scene summary.
#
# 用法: bash scripts/layout_gen_run.sh <seed> <outdir> [rooms]
#   rooms  optional; layout_gen.py decides the accepted range (2..7 today)
#
# Writes into <outdir>: spec.json, layout_report.json, checks_report.json,
# layout_gen.log, run_summary.json and -- when the render runs -- build/,
# scene/, validation_report.json.  Stale copies of the stage reports are
# removed at the start of the run, and run_summary.json records the run id, so
# a fresh summary can never describe an old run's artifacts.
#
# Exit codes (the failing stage propagates; run_summary.json carries the same
# number in "exit_code" with the stage break-down in "stages" / "ok"):
#   0  every requested stage succeeded (bundle_ok)
#   2  layout generation failed (no spec.json)
#   3  placement checks failed
#   4  render requested but not completed: pipeline failure, or still short of
#      memory after the one reduced-profile retry (pending_retry)
#   5  render ok but validation failed at the requested resolution
#   6  the visual/rule review (PANOWORLD_REVIEW=1) failed
#
# The summary grades the stages separately so a batch can tell a usable scene
# package from a layout-only run:
#   stages: layout / checks / render / validation [/ review]
#   ok:     layout_ok, render_ok, validation_ok, bundle_ok [, review_ok]
# bundle_ok = layout_ok when the render was not requested, layout_ok +
# render_ok + validation_ok when it was -- i.e. "this run produced what it
# was asked for" -- plus review_ok whenever the review was enabled.
#
# Environment:
#   PANOWORLD_RES / PANOWORLD_SAMPLES   render settings (default 1024x512, 32)
#   PANOWORLD_SKIP_RENDER=1             layout + checks only
#   PANOWORLD_MIN_FREE_MB               free memory required before rendering
#                                       (default 2200 MB: a 2048x1024 Cycles
#                                       process peaks near 1.9 GB on this host)
#   PANOWORLD_WAIT_FOR_MEMORY=1         wait for that headroom instead of
#                                       skipping, up to PANOWORLD_MEM_TIMEOUT_S
#   PANOWORLD_RETRY_MEMORY=0            disable the reduced-profile retry that
#                                       follows a memory failure (default: on)
#   PANOWORLD_RETRY_RES / _SAMPLES      reduced profile (default: half width
#                                       and height, min(samples, 16))
#   PANOWORLD_RUN_ID                    run id recorded in run_summary.json
#                                       (batch_generate.sh mints one per seed)
#   PANOWORLD_REVIEW=1                  run blender_agent/review.py: a spec
#                                       review before the render and a full
#                                       review (build + panoramas) after it;
#                                       the summary gains stages.review and
#                                       ok.review_ok, bundle_ok needs it, and a
#                                       failing review exits 6
#   PANOWORLD_REVIEW_REPAIR=1           when the review fails, apply its spec
#                                       tweaks once (furnish an empty room,
#                                       move a camera) and re-run the stages
#                                       that depend on the spec
#   PANOWORLD_REVIEW_STRICT=1           warnings fail the review as well
#   REVIEW_CMD                          external reviewer handed the panorama
#                                       paths (prints a JSON verdict)
#   PANOWORLD_PIPELINE_SH               pipeline script to run (default
#                                       scripts/blender_agent_run.sh; tests
#                                       substitute a fake)
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
PIPELINE_SH="${PANOWORLD_PIPELINE_SH:-${REPO}/scripts/blender_agent_run.sh}"
RUN_ID="${PANOWORLD_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="${SECONDS}"
RENDER_REQUESTED=1
[[ "${SKIP_RENDER}" == "1" ]] && RENDER_REQUESTED=0
REVIEW="${PANOWORLD_REVIEW:-0}"
REVIEW_REPAIR="${PANOWORLD_REVIEW_REPAIR:-0}"

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

if [[ ! -f "${PIPELINE_SH}" ]]; then
  echo "layout_gen_run.sh: no such pipeline script: ${PIPELINE_SH}" >&2
  exit 1
fi

mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
SPEC="${OUT}/spec.json"
REPORT="${OUT}/layout_report.json"
CHECKS="${OUT}/checks_report.json"
VALIDATION="${OUT}/validation_report.json"

# A new run into an existing directory must not let an old report masquerade
# as this run's result: remove the reports this run is about to write and let
# the stage statuses below decide what is regenerated.
rm -f "${SPEC}" "${REPORT}" "${CHECKS}" "${VALIDATION}" "${OUT}/run_summary.json" \
      "${OUT}/spec.pre_review.json" \
      "${OUT}/review_report.json" "${OUT}/review_prerender.json" "${OUT}/review_repair.json" \
      "${OUT}/review_pre.log" "${OUT}/review_post.log" "${OUT}/review_repair.log"

# Headroom of this container's cgroup, in MB.
#
# A second agent renders in the same cgroup, so this decides whether starting a
# Cycles process is safe.  The cgroup counts page cache as usage; the kernel
# reclaims that before it OOM kills anything, so what matters is the anonymous
# RSS (memory.stat "rss" in cgroup v1) against the limit.  Falls back to
# memory.current / memory.max (v2) and then to the host's MemAvailable.
mem_available_mb() {
  local limit="" rss="" usage=""
  for candidate in /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/max; do
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

# half of "WxH" with a floor of 256x128, the reduced profile of the retry
half_res() {
  local w="${1%x*}" h="${1#*x}" hw hh
  hw=$(( w / 2 ))
  hh=$(( h / 2 ))
  if (( hw < 256 )); then hw=256; fi
  if (( hh < 128 )); then hh=128; fi
  echo "${hw}x${hh}"
}

default_retry_samples() {
  local samples="$1"
  if (( samples > 16 )); then samples=16; fi
  echo "${samples}"
}

echo "== layout_gen =="
echo "   seed    : ${SEED}"
echo "   out     : ${OUT}"
echo "   run id  : ${RUN_ID}"
echo "   rooms   : ${ROOMS:-<seed>}"
echo "   render  : ${RES}, ${SAMPLES} samples (skip=${SKIP_RENDER})"
echo "   review  : ${REVIEW} (repair=${REVIEW_REPAIR})"

echo
echo "--- [1/4] generate the layout ---"
GEN_ARGS=(--seed "${SEED}" --out "${SPEC}" --report "${REPORT}")
[[ -n "${ROOMS}" ]] && GEN_ARGS+=(--rooms "${ROOMS}")
set +e
"${HOSTPY}" "${REPO}/blender_agent/layout_gen.py" "${GEN_ARGS[@]}" 2>&1 | tee "${OUT}/layout_gen.log"
GEN_STATUS=${PIPESTATUS[0]}
set -e

CHECKS_STATUS="not_run"
if [[ ${GEN_STATUS} -eq 0 ]]; then
  echo
  echo "--- [2/4] placement checks ---"
  set +e
  "${HOSTPY}" "${REPO}/blender_agent/checks.py" "${SPEC}" --report "${CHECKS}"
  CHECKS_STATUS=$?
  set -e
fi

# --------------------------------------------------------------------------- #
# review (blender_agent/review.py)
#
# `pre` reviews the spec alone, so an empty room or a stand in a wall is caught
# *before* a render is paid for; `post` reviews the built package (settled
# manifest, panoramas) and is the one the summary grades.  When the review
# fails and PANOWORLD_REVIEW_REPAIR=1, review.py patches spec.json (furnish an
# empty room, move a camera) -- the patch is validated by checks.py before it is
# written -- and the stages that depend on the spec run again, once.
# --------------------------------------------------------------------------- #
REVIEW_RC=0
REVIEW_PRE_RC=0
REVIEW_STATUS="not_run"
REVIEW_PRE_STATUS="not_run"
REVIEW_REPAIR_STATUS="not_attempted"
REVIEW_REPAIR_USED=0
REPAIR_ATTEMPT=0

review_stage() { # <pre|post>
  local phase="$1"
  local report="${OUT}/review_report.json"
  local args=()
  if [[ "${phase}" == "pre" ]]; then
    report="${OUT}/review_prerender.json"
    args+=(--spec-only)
  fi
  [[ "${PANOWORLD_REVIEW_STRICT:-0}" == "1" ]] && args+=(--strict)
  echo
  echo "--- review (${phase}): blender_agent/review.py ---"
  set +e
  "${HOSTPY}" "${REPO}/blender_agent/review.py" "${OUT}" --json "${report}" "${args[@]}" \
      > "${OUT}/review_${phase}.log" 2>&1
  local status=$?
  set -e
  sed 's/^/    /' "${OUT}/review_${phase}.log" || true
  if [[ "${phase}" == "pre" ]]; then
    REVIEW_PRE_RC=${status}
    if (( status == 0 )); then REVIEW_PRE_STATUS="ok"
    elif (( status == 2 )); then REVIEW_PRE_STATUS="error"
    else REVIEW_PRE_STATUS="failed"; fi
  else
    REVIEW_RC=${status}
    if (( status == 0 )); then REVIEW_STATUS="ok"
    elif (( status == 2 )); then REVIEW_STATUS="error"
    else REVIEW_STATUS="failed"; fi
  fi
  return 0
}

# Applies the review's spec tweaks; sets REVIEW_REPAIR_STATUS ("applied" when
# spec.json changed).  Never fails the script.
review_repair_stage() { # <pre|post>
  [[ -f "${SPEC}" ]] || return 0
  local report="${OUT}/review_report.json"
  [[ "$1" == "pre" ]] && report="${OUT}/review_prerender.json"
  echo
  echo "--- review repair (${1}): review.py --apply-repair ---"
  set +e
  "${HOSTPY}" "${REPO}/blender_agent/review.py" "${OUT}" --apply-repair --report "${report}" \
      > "${OUT}/review_repair.log" 2>&1
  local status=$?
  set -e
  sed 's/^/    /' "${OUT}/review_repair.log" || true
  if (( status == 0 )); then
    REVIEW_REPAIR_STATUS="applied"
  else
    REVIEW_REPAIR_STATUS="nothing_to_do"
  fi
  return 0
}

# The checks a repaired spec must pass before it is rendered again.
rerun_checks() {
  set +e
  "${HOSTPY}" "${REPO}/blender_agent/checks.py" "${SPEC}" --report "${CHECKS}"
  CHECKS_STATUS=$?
  set -e
  echo "    checks after repair: ${CHECKS_STATUS}"
  return 0
}

if [[ "${REVIEW}" == "1" && ${GEN_STATUS} -eq 0 && "${CHECKS_STATUS}" == "0" ]]; then
  review_stage pre
  if [[ "${REVIEW_PRE_STATUS}" == "failed" && "${REVIEW_REPAIR}" == "1" ]]; then
    review_repair_stage pre
    if [[ "${REVIEW_REPAIR_STATUS}" == "applied" ]]; then
      REVIEW_REPAIR_USED=1
      REPAIR_ATTEMPT=$(( REPAIR_ATTEMPT + 1 ))
      rerun_checks
      review_stage pre
    fi
  fi
fi

# Render phase.  Each attempt appends a row to ATTEMPTS_TSV:
#   attempt <TAB> resolution <TAB> samples <TAB> free_mb|"-" <TAB> exit|"-" <TAB> status
ATTEMPTS_TSV=""
LAST_EXIT=0
LAST_STATUS=""
RETRY_USED=0

append_attempt() {
  local line="$1	$2	$3	$4	$5	$6"
  if [[ -n "${ATTEMPTS_TSV}" ]]; then
    ATTEMPTS_TSV+=$'\n'
  fi
  ATTEMPTS_TSV+="${line}"
}

# Runs one pipeline attempt; sets LAST_EXIT / LAST_STATUS and appends to
# ATTEMPTS_TSV.  Never fails the script: the caller grades the result.
attempt_render() { # <attempt> <res> <samples> <gate_mb|"">
  local attempt="$1" res="$2" samples="$3" gate="$4"
  local free_mb exit_code deadline
  free_mb="$(mem_available_mb)"
  if [[ -n "${gate}" ]] && (( free_mb < gate )) && [[ "${PANOWORLD_WAIT_FOR_MEMORY:-0}" == "1" ]]; then
    deadline=$(( SECONDS + ${PANOWORLD_MEM_TIMEOUT_S:-1800} ))
    while (( free_mb < gate && SECONDS < deadline )); do
      sleep 20
      free_mb="$(mem_available_mb)"
      echo "layout_gen_run.sh: waiting for memory (have ${free_mb} MB, need ${gate} MB)"
    done
  fi
  if [[ -n "${gate}" ]] && (( free_mb < gate )); then
    LAST_EXIT=0
    LAST_STATUS="skipped_low_memory"
    echo "layout_gen_run.sh: only ${free_mb} MB free (< ${gate} MB), skipping the render."
    echo "                  re-run later or set PANOWORLD_MIN_FREE_MB to force it."
    append_attempt "${attempt}" "${res}" "${samples}" "${free_mb}" "-" "${LAST_STATUS}"
    return 0
  fi
  echo "layout_gen_run.sh: attempt ${attempt}: ${res}, ${samples} samples (free memory: ${free_mb} MB)"
  set +e
  PANOWORLD_RES="${res}" PANOWORLD_SAMPLES="${samples}" \
    bash "${PIPELINE_SH}" "${SPEC}" "${OUT}"
  exit_code=$?
  set -e
  LAST_EXIT="${exit_code}"
  if (( exit_code == 0 )); then
    LAST_STATUS="ok"
  elif (( exit_code >= 128 )); then
    LAST_STATUS="killed_signal_$(( exit_code - 128 ))"
    echo "layout_gen_run.sh: the blender pipeline died on signal $(( exit_code - 128 ))" >&2
  else
    LAST_STATUS="pipeline_exit_${exit_code}"
    echo "layout_gen_run.sh: the blender pipeline exited with ${exit_code}" >&2
  fi
  append_attempt "${attempt}" "${res}" "${samples}" "${free_mb}" "${exit_code}" "${LAST_STATUS}"
  return 0
}

RENDER_PHASE="not_attempted"
if [[ "${SKIP_RENDER}" == "1" ]]; then
  RENDER_PHASE="skipped_by_request"
elif [[ ${GEN_STATUS} -ne 0 || "${CHECKS_STATUS}" != "0" ]]; then
  RENDER_PHASE="not_attempted"
else
  RENDER_PHASE="attempts"
  echo
  echo "--- [3/4] blender pipeline (need ${MIN_FREE_MB} MB free) ---"
  attempt_render 1 "${RES}" "${SAMPLES}" "${MIN_FREE_MB}"
  RETRY_RES="${PANOWORLD_RETRY_RES:-$(half_res "${RES}")}"
  RETRY_SAMPLES="${PANOWORLD_RETRY_SAMPLES:-$(default_retry_samples "${SAMPLES}")}"
  if [[ "${PANOWORLD_RETRY_MEMORY:-1}" != "0" ]] \
     && [[ "${LAST_STATUS}" == "skipped_low_memory" || "${LAST_STATUS}" == "killed_signal_9" ]] \
     && [[ "${RETRY_RES}" != "${RES}" || "${RETRY_SAMPLES}" != "${SAMPLES}" ]]; then
    RETRY_USED=1
    echo "layout_gen_run.sh: memory failure (${LAST_STATUS}); retrying once with ${RETRY_RES}, ${RETRY_SAMPLES} samples"
    # the reduced profile is small enough to skip the headroom gate; an OOM
    # still surfaces as killed_signal_9 and ends as pending_retry
    attempt_render 2 "${RETRY_RES}" "${RETRY_SAMPLES}" ""
  fi
fi

# Post-render review of the built package, and the one repair retry: when the
# review fails on something review.py can patch, the spec changes and the render
# that produced the old build is repeated with the same profile.
if [[ "${REVIEW}" == "1" && -f "${SPEC}" ]]; then
  review_stage post
  if [[ "${REVIEW_STATUS}" == "failed" && "${REVIEW_REPAIR}" == "1" && ${REVIEW_REPAIR_USED} -eq 0 ]]; then
    review_repair_stage post
    if [[ "${REVIEW_REPAIR_STATUS}" == "applied" ]]; then
      REVIEW_REPAIR_USED=1
      rerun_checks
      if [[ "${RENDER_PHASE}" == "attempts" && "${CHECKS_STATUS}" == "0" ]]; then
        echo "layout_gen_run.sh: re-rendering the repaired spec (attempt 3)"
        attempt_render 3 "${RES}" "${SAMPLES}" "${MIN_FREE_MB}"
      fi
      review_stage post
    fi
  fi
fi

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export LGR_OUT="${OUT}" LGR_SEED="${SEED}" LGR_RUN_ID="${RUN_ID}"
export LGR_RENDER_REQUESTED="${RENDER_REQUESTED}" LGR_RENDER_PHASE="${RENDER_PHASE}"
export LGR_GEN_STATUS="${GEN_STATUS}" LGR_CHECKS_STATUS="${CHECKS_STATUS}"
export LGR_RES="${RES}" LGR_SAMPLES="${SAMPLES}"
export LGR_LAST_EXIT="${LAST_EXIT}" LGR_LAST_STATUS="${LAST_STATUS}"
export LGR_RETRY_USED="${RETRY_USED}" LGR_ATTEMPTS_TSV="${ATTEMPTS_TSV}"
export LGR_STARTED_AT="${STARTED_AT}" LGR_FINISHED_AT="${FINISHED_AT}"
export LGR_ELAPSED="$(( SECONDS - START_EPOCH ))"
export LGR_REVIEW="${REVIEW}" LGR_REVIEW_STATUS="${REVIEW_STATUS}"
export LGR_REVIEW_PRE_STATUS="${REVIEW_PRE_STATUS}" LGR_REVIEW_RC="${REVIEW_RC}"
export LGR_REVIEW_REPAIR_STATUS="${REVIEW_REPAIR_STATUS}" LGR_REVIEW_REPAIR_USED="${REVIEW_REPAIR_USED}"

set +e
"${HOSTPY}" - <<'PY'
import json, os, sys

out = os.environ["LGR_OUT"]
seed = int(os.environ["LGR_SEED"])
run_id = os.environ["LGR_RUN_ID"]
render_requested = os.environ["LGR_RENDER_REQUESTED"] == "1"
render_phase = os.environ["LGR_RENDER_PHASE"]
gen_status = int(os.environ["LGR_GEN_STATUS"])
checks_status = os.environ["LGR_CHECKS_STATUS"]
res = os.environ["LGR_RES"]
samples = int(os.environ["LGR_SAMPLES"])
last_exit = int(os.environ["LGR_LAST_EXIT"])
last_status = os.environ["LGR_LAST_STATUS"]
retry_used = os.environ["LGR_RETRY_USED"] == "1"
review_enabled = os.environ.get("LGR_REVIEW", "0") == "1"
review_status = os.environ.get("LGR_REVIEW_STATUS", "not_run")
review_pre_status = os.environ.get("LGR_REVIEW_PRE_STATUS", "not_run")
review_rc = int(os.environ.get("LGR_REVIEW_RC", "0"))
review_repair_status = os.environ.get("LGR_REVIEW_REPAIR_STATUS", "not_attempted")
review_repair_used = os.environ.get("LGR_REVIEW_REPAIR_USED", "0") == "1"

attempts = []
for line in os.environ.get("LGR_ATTEMPTS_TSV", "").splitlines():
    if not line.strip():
        continue
    attempt, a_res, a_samples, a_free, a_exit, a_status = line.split("\t")
    attempts.append({
        "attempt": int(attempt),
        "resolution": a_res,
        "samples": int(a_samples),
        "free_mb": None if a_free == "-" else int(a_free),
        "exit_code": None if a_exit == "-" else int(a_exit),
        "status": a_status,
    })

spec_path = os.path.join(out, "spec.json")
checks_path = os.path.join(out, "checks_report.json")
validation_path = os.path.join(out, "validation_report.json")

summary = {
    "kind": "layout_gen_run",
    "schema_version": 2,
    "run_id": run_id,
    "seed": seed,
    "out": out,
    "render_requested": render_requested,
    "started_at": os.environ["LGR_STARTED_AT"],
    "finished_at": os.environ["LGR_FINISHED_AT"],
    "elapsed_seconds": float(os.environ["LGR_ELAPSED"]),
    "stages": {},
    "ok": {},
    "exit_code": None,
    "failure": None,
}

# --- legacy scene facts (kept for readers of the old summary format) ------- #
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

# --- stage grading --------------------------------------------------------- #
gen_ok = gen_status == 0 and os.path.isfile(spec_path)
failed_check_names = []
if os.path.isfile(checks_path):
    with open(checks_path, encoding="utf-8") as handle:
        checks_report = json.load(handle)
    failed_check_names = [check["name"] for check in checks_report.get("checks", [])
                          if not check["pass"]]
checks_ok = checks_status == "0"
layout_ok = bool(gen_ok and checks_ok)

render_status = "not_attempted"
render_ok = None
validation_ok = None
validation = None

if not render_requested:
    render_status = "skipped_by_request"
    render_ok = None
elif render_phase == "not_attempted":
    render_status = "not_attempted"
    render_ok = False
else:
    render_ok = False
    if last_status in ("skipped_low_memory", "killed_signal_9"):
        render_status = "pending_retry"
    elif os.path.isfile(validation_path):
        with open(validation_path, encoding="utf-8") as handle:
            validation = json.load(handle)
        files_check = next((check for check in validation.get("checks", [])
                            if check.get("name") == "files_present"), None)
        files_ok = bool(files_check and files_check.get("pass"))
        render_status = "ok" if files_ok else "failed"
        render_ok = files_ok
    else:
        render_status = "failed"

if validation is None and os.path.isfile(validation_path):
    with open(validation_path, encoding="utf-8") as handle:
        validation = json.load(handle)
failed_validation_checks = []
if validation is not None:
    failed_validation_checks = [check["name"] for check in validation.get("checks", [])
                                if not check["pass"]]
    canonical = res == "2048x1024"
    pass_at_resolution = bool(validation.get("pass")) or \
        (failed_validation_checks == ["canonical_resolution"] and not canonical)
    validation_ok = bool(pass_at_resolution)
    summary["validation"] = {
        "pass": validation.get("pass"),
        "pass_at_resolution": validation_ok,
        "canonical_resolution": canonical,
        "failed": failed_validation_checks,
        "depth": [{"camera": entry["camera"],
                   "median_mm": round(entry["median_error_m"] * 1000, 3),
                   "p95_mm": round(entry["p95_error_m"] * 1000, 3),
                   "valid_fraction": round(entry["valid_fraction"], 5),
                   "door_ratio": round(entry["door_spot_check"].get("ratio", 0.0), 3)}
                  for entry in next((check for check in validation.get("checks", [])
                                     if check["name"] == "depth_matches_geometry"), {}).get("cameras", [])],
    }

if render_requested:
    bundle_ok = bool(layout_ok and render_ok is True and validation_ok is True)
else:
    bundle_ok = layout_ok

# --- review ---------------------------------------------------------------- #
review = None
review_ok = None
if review_enabled:
    report_path = os.path.join(out, "review_report.json")
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as handle:
            review = json.load(handle)
    review_ok = bool(review.get("pass")) if review is not None else False
    summary["review"] = {
        "status": review_status,
        "pass": review_ok,
        "pre_render_status": review_pre_status,
        "exit_code": review_rc,
        "report": report_path if review is not None else None,
        "scopes": (review or {}).get("scopes"),
        "score": (review or {}).get("score"),
        "issues": {"fail": ((review or {}).get("stats") or {}).get("issues_fail"),
                   "warn": ((review or {}).get("stats") or {}).get("issues_warn")},
        "failed_checks": sorted({issue["check"] for issue in (review or {}).get("issues", [])
                                 if issue.get("severity") == "fail"}),
        "issue_kinds": sorted({issue.get("kind") for issue in (review or {}).get("issues", [])
                               if issue.get("severity") == "fail" and issue.get("kind")}),
        "repair": {"used": review_repair_used, "status": review_repair_status},
    }
    bundle_ok = bool(bundle_ok and review_ok)

# --- failure / exit code --------------------------------------------------- #
failure = None
if not gen_ok:
    detail = f"layout_gen.py exited {gen_status}"
    log_path = os.path.join(out, "layout_gen.log")
    if os.path.isfile(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        for line in reversed(lines):
            if "FAILED" in line or "Error" in line or "error" in line:
                detail = line
                break
    failure = {"stage": "layout", "kind": "generation_failed", "detail": detail}
elif not checks_ok:
    failure = {"stage": "checks", "kind": "checks_failed",
               "detail": f"checks.py exited {checks_status}",
               "checks": failed_check_names}
elif render_requested and render_status == "pending_retry":
    failure = {"stage": "render", "kind": last_status,
               "detail": f"render incomplete after {len(attempts)} attempt(s)"}
elif render_requested and render_status == "failed":
    kind = last_status if last_status not in ("ok", "") else "artifacts_missing"
    if last_status == "ok":
        files = next((check.get("missing", []) for check in (validation or {}).get("checks", [])
                      if check.get("name") == "files_present"), [])
        detail = "render finished but the viewpoint set is incomplete" + \
                 (f": {', '.join(map(str, files[:5]))}" if files else "")
    else:
        detail = f"the blender pipeline failed ({last_status})"
    failure = {"stage": "render", "kind": kind, "detail": detail}
elif render_requested and validation_ok is False:
    failure = {"stage": "validation", "kind": "validation_failed",
               "detail": ", ".join(failed_validation_checks) or "validation failed",
               "checks": failed_validation_checks}
elif review_enabled and review_ok is False:
    kinds = ((summary.get("review") or {}).get("issue_kinds") or [])
    failure = {"stage": "review",
               "kind": "review_error" if review_status == "error" else "review_failed",
               "detail": ", ".join(kinds) or "the review reported failures",
               "checks": (summary.get("review") or {}).get("failed_checks") or []}

if failure is None:
    exit_code = 0
elif failure["stage"] == "layout":
    exit_code = 2
elif failure["stage"] == "checks":
    exit_code = 3
elif failure["stage"] == "render":
    exit_code = 4
elif failure["stage"] == "validation":
    exit_code = 5
else:
    exit_code = 6

summary["failure"] = failure
summary["exit_code"] = exit_code
summary["render"] = {
    "status": render_status,
    "resolution": res,
    "samples": samples,
    "pipeline_exit_code": last_exit,
    "attempts": attempts,
    "retry_used": retry_used,
}
summary["checks"] = {"pass": checks_ok,
                     "status": "ok" if checks_ok else ("not_run" if checks_status == "not_run" else "failed"),
                     "failed": failed_check_names}
summary["stages"] = {
    "layout": "ok" if layout_ok else "failed",
    "checks": "ok" if checks_ok else ("not_run" if checks_status == "not_run" else "failed"),
    "render": render_status,
    "validation": "ok" if validation_ok else ("not_run" if validation_ok is None else "failed"),
}
summary["ok"] = {
    "layout_ok": bool(layout_ok),
    "render_ok": render_ok,
    "validation_ok": validation_ok,
    "bundle_ok": bool(bundle_ok),
}
if review_enabled:
    summary["stages"]["review"] = review_status
    summary["ok"]["review_ok"] = review_ok

with open(os.path.join(out, "run_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

validation = summary.get("validation")
if validation is None:
    validation_note = "validation=n/a (not run)"
else:
    validation_note = ("validation=%s (at %s: %s%s)"
                       % (validation.get("pass"), res,
                          "PASS" if validation.get("pass_at_resolution") else "FAIL",
                          (", failed %s" % validation["failed"]) if validation.get("failed") else ""))
review_note = ""
if review_enabled:
    review_block = summary.get("review") or {}
    review_note = (" review=%s (%s fail / %s warn) "
                   % ("PASS" if review_block.get("pass") else review_block.get("status", "FAIL"),
                      (review_block.get("issues") or {}).get("fail"),
                      (review_block.get("issues") or {}).get("warn")))
print()
print(f"layout_gen: seed {seed}: "
      f"layout={'PASS' if layout_ok else 'FAIL'} "
      f"render={render_status} "
      f"{validation_note} "
      f"{review_note}"
      f"bundle={'PASS' if bundle_ok else 'FAIL'} "
      f"run_id={run_id}")
if failure is not None:
    print(f"layout_gen: failure: {failure['stage']}:{failure['kind']}: {failure['detail']}")
raise SystemExit(exit_code)
PY
EXIT_CODE=$?
set -e

echo
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "layout_gen finished: ${OUT}"
else
  echo "layout_gen finished: ${OUT} (failed, exit ${EXIT_CODE})" >&2
fi
exit "${EXIT_CODE}"
