"""Structural diversity of generated houses -- and the 100-seed benchmark.

`layout_gen.py` can produce a thousand specs; the question this tool answers is
whether they are a thousand *different* houses or the same corridor with other
furniture.  It reads specs (a batch root, explicit paths, or seeds it generates)
and measures, per scene:

| metric | of which |
|---|---|
| `rooms`, `room_types`, `floors` | room count, type histogram, storey count |
| `structure` | the **typed adjacency graph**: edges between room *types*, its cycle rank, and a hash that ignores furniture, size and seed |
| `shape` | footprint bounding box, area, aspect ratio |
| `functions` | which room types the plan covers |
| `furniture` | item count, distinct assets, parts/table densities, per-room-type asset sets, combo hash |
| `cameras` | per-room counts, nearest-neighbour spacing, rooms without a viewpoint |
| `lighting` | sun elevation/azimuth, sky strength, exposure, windows |

and then aggregates them: histograms, per-family counts, and **topology
repeats** -- scenes whose `structure_hash` is identical.  Two houses that are
the same typed graph with different furniture are one topology, not two, and
that is exactly what the `duplicate_rate` says.  `plan_hash` is the stricter
neighbour (structure + coarse shape bins) for "near-identical plan" detection.

Command line::

    # explicit specs / batch roots / layout_report.json files
    python tools/diversity_report.py specs outputs/stream/*/spec.json --out /tmp/diversity.json

    # generate in process and measure (no files written unless --out is given)
    python tools/diversity_report.py seeds --seeds 0-99 --out /tmp/diversity100.json

    # the batch benchmark: 100 seeds of layout+diversity, a stratified sample
    # rendered and reviewed, collated into one report
    python tools/diversity_report.py benchmark --seeds 0-99 --sample 10 \
        --root outputs/benchmark100

The `benchmark` mode drives the real scripts (`scripts/batch_generate.sh`), so
its timings are the pipeline's own and its failure distribution comes from the
same grading the batch prints: `layout_ok` / `render_ok` / `validation_ok` /
`review_ok` / `bundle_ok`.  Rendering is the expensive part, so the sample is
drawn deliberately: at least one scene per topology family, then seeds that add
a *new* structure hash, filled up in seed order.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import math
import os

import statistics
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_BATCH = os.path.join(REPO_ROOT, "scripts", "batch_generate.sh")
DEFAULT_HOSTPY = os.environ.get("PANOWORLD_HOSTPY", "/opt/conda/envs/diffaero/bin/python")

TRANSIT_TYPES = ("corridor", "balcony", "entry")


# --------------------------------------------------------------------------- #
# spec reading / small helpers
# --------------------------------------------------------------------------- #
def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _round(value: float, step: float) -> float:
    return round(value / step) * step


def load_specs(paths: Sequence[str]) -> List[dict]:
    """Expand paths/dirs/globs into ``[{"path", "spec", "layout_report"}]``."""
    out: List[dict] = []
    seen = set()
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "spec.json"))))
            files.extend(sorted(glob.glob(os.path.join(path, "scene_*", "spec.json"))))
        else:
            files.extend(sorted(glob.glob(path)) or [path])
    for path in files:
        path = os.path.abspath(path)
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            with open(path, encoding="utf-8") as handle:
                spec = json.load(handle)
        except Exception:
            continue
        report_path = os.path.join(os.path.dirname(path), "layout_report.json")
        report = None
        if os.path.isfile(report_path):
            try:
                with open(report_path, encoding="utf-8") as handle:
                    report = json.load(handle)
            except Exception:
                report = None
        out.append({"path": path, "spec": spec, "layout_report": report})
    return out


def parse_seeds(text: str) -> List[int]:
    """``"0-99"`` / ``"0,7,13"`` / ``"5-9,20"`` -> sorted ints."""
    seeds: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            low, _, high = chunk.partition("-")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(chunk))
    return sorted(dict.fromkeys(seeds))


# --------------------------------------------------------------------------- #
# per-scene metrics
# --------------------------------------------------------------------------- #
def scene_metrics(spec: dict, seed: Optional[int] = None, layout_report: Optional[dict] = None,
                  path: Optional[str] = None) -> dict:
    """Structure, shape, function, furniture, camera and lighting numbers."""
    rooms = spec.get("rooms") or []
    room_type = {room["id"]: room.get("type", "unknown") for room in rooms}
    types = [room.get("type", "unknown") for room in rooms]

    edges: List[Tuple[str, str]] = []
    for opening in spec.get("openings") or []:
        between = opening.get("between") or []
        if len(between) != 2:
            continue
        first, second = room_type.get(between[0]), room_type.get(between[1])
        if first is None or second is None or between[0] == between[1]:
            continue
        edges.append(tuple(sorted((first, second))))
    unique_edges = sorted(set(edges))

    adjacency: Dict[str, set] = {room["id"]: set() for room in rooms}
    for opening in spec.get("openings") or []:
        between = opening.get("between") or []
        if len(between) == 2 and between[0] in adjacency and between[1] in adjacency:
            adjacency[between[0]].add(between[1])
            adjacency[between[1]].add(between[0])
    components = 0
    unvisited = set(adjacency)
    while unvisited:
        components += 1
        stack = [unvisited.pop()]
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour in unvisited:
                    unvisited.discard(neighbour)
                    stack.append(neighbour)
    cycles = max(0, len(unique_edges) - len(rooms) + components)

    points = [point for room in rooms for point in room.get("polygon") or []]
    if points:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        width, depth = max(xs) - min(xs), max(ys) - min(ys)
        area = round(width * depth, 2)
        aspect = round(max(width, depth) / max(1e-6, min(width, depth)), 2)
    else:
        width = depth = area = 0.0
        aspect = 1.0

    type_counts: Dict[str, int] = {}
    for name in types:
        type_counts[name] = type_counts.get(name, 0) + 1

    furniture_by_room: Dict[str, List[str]] = {}
    for item in spec.get("furniture") or []:
        furniture_by_room.setdefault(item.get("room", "?"), []).append(str(item.get("asset")))
    combo: Dict[str, List[str]] = {}
    for room in rooms:
        assets = sorted(furniture_by_room.get(room["id"], []))
        combo.setdefault(room.get("type", "unknown"), []).extend(assets)
    combo = {room_type_name: sorted(assets) for room_type_name, assets in combo.items()}

    cameras = spec.get("cameras") or []
    per_room: Dict[str, int] = {}
    for camera in cameras:
        per_room[camera.get("room", "?")] = per_room.get(camera.get("room", "?"), 0) + 1
    positions = [camera.get("pos", [0.0, 0.0]) for camera in cameras]
    gaps: List[float] = []
    for i in range(len(positions)):
        best = float("inf")
        for j in range(len(positions)):
            if i == j:
                continue
            best = min(best, math.hypot(positions[i][0] - positions[j][0],
                                        positions[i][1] - positions[j][1]))
        if best < float("inf"):
            gaps.append(best)

    style = spec.get("style") or {}
    structure = {"rooms": type_counts, "edges": unique_edges, "cycles": cycles}
    plan = dict(structure, aspect=_round(aspect, 0.5), area=_round(area, 10.0))
    return {
        "seed": seed,
        "path": path,
        "family": (layout_report or {}).get("family") or (layout_report or {}).get("shape"),
        "rooms": len(rooms),
        "room_types": type_counts,
        "floors": len(spec.get("floors") or []) or 1,
        "structure": structure,
        "structure_hash": _hash(structure),
        "plan_hash": _hash(plan),
        "shape": {"width_m": round(width, 2), "depth_m": round(depth, 2), "area_m2": area,
                  "aspect": aspect},
        "functions": sorted(type_counts),
        "furniture": {"count": len(spec.get("furniture") or []),
                      "distinct_assets": len({item.get("asset") for item in spec.get("furniture") or []}),
                      "combo_hash": _hash(combo), "per_room_type": combo},
        "cameras": {"count": len(cameras), "per_room": per_room,
                    "rooms_without_camera": sorted(room["id"] for room in rooms
                                                   if room["id"] not in per_room
                                                   and room.get("type") not in TRANSIT_TYPES),
                    "min_gap_m": round(min(gaps), 3) if gaps else None,
                    "mean_gap_m": round(statistics.fmean(gaps), 3) if gaps else None},
        "lighting": {"sun_elevation_deg": style.get("sun_elevation_deg"),
                     "sun_azimuth_deg": style.get("sun_azimuth_deg"),
                     "sky_strength": style.get("sky_strength"),
                     "exposure": style.get("exposure"),
                     "windows": len(spec.get("windows") or [])},
    }


def _histogram(values: Iterable[float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda entry: float(entry[0])))


def diversity_summary(metrics: Sequence[dict]) -> dict:
    """Aggregate per-scene metrics: histograms, coverage and topology repeats."""
    if not metrics:
        return {"count": 0}

    types: Dict[str, int] = {}
    families: Dict[str, int] = {}
    for entry in metrics:
        families[str(entry.get("family"))] = families.get(str(entry.get("family")), 0) + 1
        for name in entry["functions"]:
            types[name] = types.get(name, 0) + 1

    groups: Dict[str, List[dict]] = {}
    plan_groups: Dict[str, List[dict]] = {}
    for entry in metrics:
        groups.setdefault(entry["structure_hash"], []).append(entry)
        plan_groups.setdefault(entry["plan_hash"], []).append(entry)

    def repeat_report(bucket: Dict[str, List[dict]]) -> dict:
        repeated = {key: entries for key, entries in bucket.items() if len(entries) > 1}
        top = sorted(repeated.items(), key=lambda item: -len(item[1]))[:5]
        return {
            "unique": len(bucket),
            "duplicate_rate": round(1.0 - len(bucket) / float(len(metrics)), 4),
            "groups": [
                {"hash": key,
                 "count": len(entries),
                 "seeds": sorted([entry["seed"] for entry in entries
                                  if entry.get("seed") is not None])[:20],
                 "families": sorted({str(entry.get("family")) for entry in entries}),
                 "furniture_combos": len({entry["furniture"]["combo_hash"] for entry in entries}),
                 "note": ("same typed topology, different furniture"
                          if len({entry["furniture"]["combo_hash"] for entry in entries}) > 1
                          else "same topology and the same furniture sets")}
                for key, entries in top
            ],
            "repeat_scenes": sum(len(entries) for entries in repeated.values()),
        }

    furniture_metrics = [entry["furniture"] for entry in metrics]
    camera_counts = [entry["cameras"]["count"] for entry in metrics]
    room_counts = [entry["rooms"] for entry in metrics]
    aspects = [entry["shape"]["aspect"] for entry in metrics]
    areas = [entry["shape"]["area_m2"] for entry in metrics]
    sun_azimuths = [entry["lighting"]["sun_azimuth_deg"] for entry in metrics
                    if entry["lighting"]["sun_azimuth_deg"] is not None]
    camera_gaps = [entry["cameras"]["min_gap_m"] for entry in metrics
                   if entry["cameras"]["min_gap_m"] is not None]
    return {
        "count": len(metrics),
        "families": dict(sorted(families.items())),
        "rooms": {"histogram": _histogram(room_counts), "min": min(room_counts),
                  "max": max(room_counts), "mean": round(statistics.fmean(room_counts), 2)},
        "floors": _histogram([entry["floors"] for entry in metrics]),
        "cycles": {"histogram": _histogram([entry["structure"]["cycles"] for entry in metrics]),
                   "max": max(entry["structure"]["cycles"] for entry in metrics)},
        "function_coverage": dict(sorted(types.items(), key=lambda entry: -entry[1])),
        "shape": {"aspect_min": min(aspects), "aspect_max": max(aspects),
                  "aspect_mean": round(statistics.fmean(aspects), 2),
                  "area_min_m2": min(areas), "area_max_m2": max(areas),
                  "area_mean_m2": round(statistics.fmean(areas), 2)},
        "furniture": {
            "count_min": min(entry["count"] for entry in furniture_metrics),
            "count_max": max(entry["count"] for entry in furniture_metrics),
            "count_mean": round(statistics.fmean([entry["count"] for entry in furniture_metrics]), 2),
            "distinct_assets_mean": round(statistics.fmean(
                [entry["distinct_assets"] for entry in furniture_metrics]), 2),
        },
        "cameras": {"count_min": min(camera_counts), "count_max": max(camera_counts),
                    "count_mean": round(statistics.fmean(camera_counts), 2),
                    "rooms_without_camera": sum(len(entry["cameras"]["rooms_without_camera"])
                                                for entry in metrics),
                    "min_gap_m_min": min(camera_gaps) if camera_gaps else None},
        "lighting": {"sun_azimuth_distinct": len(set(sun_azimuths)),
                     "sun_azimuth_values": sorted(set(sun_azimuths))[:12]},
        "structures": repeat_report(groups),
        "plans": repeat_report(plan_groups),
    }


# --------------------------------------------------------------------------- #
# benchmark: layout sweep -> diversity -> stratified sample -> render + review
# --------------------------------------------------------------------------- #
def _run_batch(start: int, count: int, root: str, env_overrides: Dict[str, str],
               seeds: Optional[Sequence[int]] = None) -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("PANOWORLD_")}
    env.update(env_overrides)
    if seeds is not None:
        env["PANOWORLD_BATCH_SEEDS"] = ",".join(str(seed) for seed in seeds)
    started = datetime.datetime.now(datetime.timezone.utc)
    os.makedirs(root, exist_ok=True)
    log_path = os.path.join(root, "batch_run.log")
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(["bash", DEFAULT_BATCH, str(start), str(count), root],
                                env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    summary_path = os.path.join(root, "batch_summary.json")
    summary = None
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    return {
        "exit_code": result.returncode,
        "seconds": round((datetime.datetime.now(datetime.timezone.utc) - started).total_seconds(), 2),
        "summary": summary,
        "summary_path": summary_path if summary is not None else None,
        "log": log_path,
    }


def _blender_running() -> List[str]:
    try:
        result = subprocess.run(["pgrep", "-a", "blender"], capture_output=True, text=True)
    except Exception:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def choose_sample(metrics: Sequence[dict], size: int) -> dict:
    """Stratified sample: one per family, then new structures, then seed order."""
    by_family: Dict[str, List[dict]] = {}
    for entry in metrics:
        by_family.setdefault(str(entry.get("family")), []).append(entry)
    for entries in by_family.values():
        entries.sort(key=lambda entry: (entry.get("seed") if entry.get("seed") is not None else -1))

    chosen: List[dict] = []
    reasons: Dict[int, str] = {}
    for family in sorted(by_family):
        if len(chosen) >= size:
            break
        entry = by_family[family][0]
        chosen.append(entry)
        reasons[entry["seed"]] = f"first of family '{family}'"

    seen_structures = {entry["structure_hash"] for entry in chosen}
    for entry in sorted(metrics, key=lambda item: item.get("seed") if item.get("seed") is not None else -1):
        if len(chosen) >= size:
            break
        if entry in chosen or entry["structure_hash"] in seen_structures:
            continue
        seen_structures.add(entry["structure_hash"])
        chosen.append(entry)
        reasons[entry["seed"]] = "new topology structure"
    for entry in sorted(metrics, key=lambda item: item.get("seed") if item.get("seed") is not None else -1):
        if len(chosen) >= size:
            break
        if entry in chosen:
            continue
        chosen.append(entry)
        reasons[entry["seed"]] = "fill"
    chosen.sort(key=lambda entry: entry["seed"])
    return {
        "seeds": [entry["seed"] for entry in chosen],
        "families": {str(entry["seed"]): entry.get("family") for entry in chosen},
        "reasons": {str(seed): reason for seed, reason in reasons.items()},
        "per_family": {family: sum(1 for entry in chosen if str(entry.get("family")) == family)
                       for family in sorted({str(entry.get("family")) for entry in metrics})},
    }


def _per_scene_costs(root: str, seeds: Sequence[int]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for seed in seeds:
        scene = os.path.join(root, "scene_%04d" % seed)
        summary_path = os.path.join(scene, "run_summary.json")
        if not os.path.isfile(summary_path):
            continue
        try:
            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
        except Exception:
            continue
        review = summary.get("review") or {}
        out[str(seed)] = {
            "seconds": summary.get("elapsed_seconds"),
            "status": "bundle_ok" if (summary.get("ok") or {}).get("bundle_ok") else "failed",
            "render": (summary.get("render") or {}).get("status"),
            "review": review.get("status"),
            "review_pass": review.get("pass"),
            "review_issues": review.get("issues"),
            "review_failed_checks": review.get("failed_checks"),
            "repair": review.get("repair"),
            "resolution": (summary.get("render") or {}).get("resolution"),
        }
    return out


def benchmark(args) -> int:
    root = os.path.abspath(args.root)
    seeds = parse_seeds(args.seeds)
    if not seeds:
        print("benchmark: --seeds is empty", file=sys.stderr)
        return 2
    os.makedirs(root, exist_ok=True)
    started = _utc_now()
    report: dict = {
        "kind": "panoworld_benchmark",
        "schema_version": 1,
        "started_at": started,
        "root": root,
        "seeds": {"range": [seeds[0], seeds[-1]], "count": len(seeds)},
        "config": {"sample": args.sample, "shape": args.shape, "resolution": args.resolution,
                   "samples": args.samples, "min_free_mb": args.min_free_mb,
                   "review": True, "repair": bool(args.repair),
                   "layout_batch": os.path.join(root, "layout"),
                   "sample_batch": os.path.join(root, "sample")},
    }

    # -- 1: layout over every seed ------------------------------------------ #
    print(f"[1/4] layout sweep: {len(seeds)} seeds -> {os.path.join(root, 'layout')}")
    layout_env = {"PANOWORLD_BATCH_RESUME": "1" if args.resume else "0",
                  "PANOWORLD_LAYOUT_SHAPE": args.shape}
    if args.rooms:
        layout_env["PANOWORLD_ROOMS"] = str(args.rooms)
    layout = _run_batch(seeds[0], len(seeds), report["config"]["layout_batch"], layout_env,
                        seeds=seeds)
    layout_summary = layout["summary"] or {}
    report["layout"] = {
        "seconds": layout["seconds"],
        "exit_code": layout["exit_code"],
        "levels": layout_summary.get("levels"),
        "passed": layout_summary.get("passed"),
        "failed": layout_summary.get("failed"),
        "failure_reasons": layout_summary.get("failure_reasons"),
        "batch_summary": layout["summary_path"],
        "log": layout["log"],
    }
    ok_seeds = [row["seed"] for row in layout_summary.get("scenes", [])
                if (row.get("ok") or {}).get("layout_ok")]
    layout_seconds = [entry["seconds"] for entry in
                      _per_scene_costs(report["config"]["layout_batch"], seeds).values()
                      if entry.get("seconds")]
    if layout_seconds:
        report["layout"]["per_scene_seconds"] = {
            "mean": round(statistics.fmean(layout_seconds), 2),
            "median": round(statistics.median(layout_seconds), 2),
            "max": max(layout_seconds)}
    print(f"      layout_ok {layout_summary.get('levels', {}).get('layout_ok')}/{len(seeds)}"
          f" in {layout['seconds']:.1f} s"
          f" (~{report['layout'].get('per_scene_seconds', {}).get('mean')} s/scene)")

    # -- 2: diversity ------------------------------------------------------- #
    print("[2/4] diversity metrics")
    specs = load_specs([os.path.join(report["config"]["layout_batch"], "scene_*", "spec.json")])
    metrics = []
    for entry in specs:
        seed = None
        if os.path.basename(os.path.dirname(entry["path"])).startswith("scene_"):
            try:
                seed = int(os.path.basename(os.path.dirname(entry["path"]))[len("scene_"):])
            except ValueError:
                seed = None
        metrics.append(scene_metrics(entry["spec"], seed=seed,
                                     layout_report=entry["layout_report"], path=entry["path"]))
    summary = diversity_summary(metrics)
    report["diversity"] = dict(summary, scenes=metrics)
    with open(os.path.join(root, "diversity.json"), "w", encoding="utf-8") as handle:
        json.dump({"kind": "panoworld_diversity", "created_utc": _utc_now(),
                   "count": len(metrics), "summary": summary, "scenes": metrics},
                  handle, indent=2)
    print(f"      {summary.get('count')} structures: {summary.get('structures', {}).get('unique')} "
          f"unique (duplicate rate {summary.get('structures', {}).get('duplicate_rate')}), "
          f"plan duplicates {summary.get('plans', {}).get('duplicate_rate')}")

    # -- 3: the stratified sample ------------------------------------------ #
    sample = choose_sample(metrics, max(1, args.sample))
    report["sample"] = sample
    print(f"[3/4] sample: {sample['seeds']} (per family {sample['per_family']})")

    # -- 4: render + review the sample ------------------------------------- #
    if args.no_render:
        report["render"] = {"status": "skipped_by_request"}
    else:
        running = _blender_running()
        if running and not args.force:
            print("benchmark: another blender process is running:\n  " + "\n  ".join(running[:5]),
                  file=sys.stderr)
            print("benchmark: refusing to start renders (use --force to override)", file=sys.stderr)
            report["render"] = {"status": "blocked_by_concurrent_blender", "processes": running[:5]}
        else:
            print(f"[4/4] render + review {len(sample['seeds'])} scenes -> "
                  f"{report['config']['sample_batch']}")
            render_env = {"PANOWORLD_BATCH_RENDER": "1", "PANOWORLD_REVIEW": "1",
                          "PANOWORLD_BATCH_RESUME": "1" if args.resume else "0",
                          "PANOWORLD_RES": args.resolution, "PANOWORLD_SAMPLES": str(args.samples),
                          "PANOWORLD_MIN_FREE_MB": str(args.min_free_mb),
                          "PANOWORLD_LAYOUT_SHAPE": args.shape}
            if args.repair:
                render_env["PANOWORLD_REVIEW_REPAIR"] = "1"
            if args.rooms:
                render_env["PANOWORLD_ROOMS"] = str(args.rooms)
            rendered = _run_batch(seeds[0], len(sample["seeds"]), report["config"]["sample_batch"],
                                  render_env, seeds=sample["seeds"])
            rendered_summary = rendered["summary"] or {}
            costs = _per_scene_costs(report["config"]["sample_batch"], sample["seeds"])
            seconds = [entry["seconds"] for entry in costs.values() if entry.get("seconds")]
            report["render"] = {
                "seconds": rendered["seconds"],
                "exit_code": rendered["exit_code"],
                "levels": rendered_summary.get("levels"),
                "passed": rendered_summary.get("passed"),
                "pending_retry": rendered_summary.get("pending_retry"),
                "failed": rendered_summary.get("failed"),
                "failure_reasons": rendered_summary.get("failure_reasons"),
                "failed_checks": rendered_summary.get("failed_checks"),
                "review_issues": rendered_summary.get("review_issues"),
                "per_scene": costs,
                "per_scene_seconds": {"mean": round(statistics.fmean(seconds), 1) if seconds else None,
                                      "median": round(statistics.median(seconds), 1) if seconds else None,
                                      "max": max(seconds) if seconds else None},
                "batch_summary": rendered["summary_path"],
                "log": rendered["log"],
            }
            print(f"      bundle_ok {rendered_summary.get('passed')}/{len(sample['seeds'])}, "
                  f"review_ok {(rendered_summary.get('levels') or {}).get('review_ok')}"
                  f"/{len(sample['seeds'])} in {rendered['seconds']:.1f} s"
                  f" (~{report['render']['per_scene_seconds']['mean']} s/scene)")

    report["finished_at"] = _utc_now()
    report_path = os.path.join(root, "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"benchmark report: {report_path}")
    layout_ok = report["layout"].get("passed") == len(seeds)
    render = report.get("render") or {}
    render_ok = render.get("status") is None and render.get("passed") == len(sample["seeds"])
    return 0 if (layout_ok and (args.no_render or render_ok)) else 1


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def _write(path: Optional[str], payload: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Structural diversity of generated houses")
    sub = parser.add_subparsers(dest="command", required=True)

    specs_cmd = sub.add_parser("specs", help="measure existing spec files / batch roots")
    specs_cmd.add_argument("paths", nargs="+")
    specs_cmd.add_argument("--out", default=None)
    specs_cmd.add_argument("--scenes", action="store_true", help="keep the per-scene metrics")

    seeds_cmd = sub.add_parser("seeds", help="generate specs in process and measure them")
    seeds_cmd.add_argument("--seeds", default="0-99")
    seeds_cmd.add_argument("--shape", default="auto")
    seeds_cmd.add_argument("--rooms", type=int, default=None)
    seeds_cmd.add_argument("--out", default=None)
    seeds_cmd.add_argument("--scenes", action="store_true")
    seeds_cmd.add_argument("--hostpy", default=DEFAULT_HOSTPY,
                           help="interpreter that generates the specs")

    bench_cmd = sub.add_parser("benchmark", help="layout sweep + stratified render/review sample")
    bench_cmd.add_argument("--seeds", default="0-99")
    bench_cmd.add_argument("--sample", type=int, default=10, help="scenes to render (8-12)")
    bench_cmd.add_argument("--root", default=os.path.join(REPO_ROOT, "outputs", "benchmark100"))
    bench_cmd.add_argument("--shape", default="auto")
    bench_cmd.add_argument("--rooms", type=int, default=None)
    bench_cmd.add_argument("--resolution", default="1024x512")
    bench_cmd.add_argument("--samples", type=int, default=24)
    bench_cmd.add_argument("--min-free-mb", type=int, default=2200,
                           help="layout_gen_run.sh render gate (cgroup headroom)")
    bench_cmd.add_argument("--repair", action="store_true", help="PANOWORLD_REVIEW_REPAIR=1")
    bench_cmd.add_argument("--resume", action="store_true", help="reuse completed scenes")
    bench_cmd.add_argument("--no-render", action="store_true")
    bench_cmd.add_argument("--force", action="store_true",
                           help="render even when another blender process is running")

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        return benchmark(args)

    if args.command == "specs":
        entries = load_specs(args.paths)
        metrics = [scene_metrics(entry["spec"], layout_report=entry["layout_report"],
                                 path=entry["path"]) for entry in entries]
    else:
        seeds = parse_seeds(args.seeds)
        generated = generate_specs(seeds, args.hostpy, args.shape, args.rooms)
        metrics = [scene_metrics(spec, seed=seed, layout_report=report)
                   for seed, spec, report in generated]

    summary = diversity_summary(metrics)
    payload = {"kind": "panoworld_diversity", "created_utc": _utc_now(),
               "count": len(metrics), "summary": summary}
    if args.scenes:
        payload["scenes"] = metrics
    if args.out:
        _write(args.out, payload)
    _print_summary(summary)
    return 0


def generate_specs(seeds: Sequence[int], hostpy: str, shape: str,
                   rooms: Optional[int] = None) -> List[Tuple[int, dict, Optional[dict]]]:
    """Write each seed's spec through the generator CLI and read it back."""
    import tempfile

    out: List[Tuple[int, dict, Optional[dict]]] = []
    tmpdir = tempfile.mkdtemp(prefix="diversity_")
    for seed in seeds:
        path = os.path.join(tmpdir, f"seed_{seed}.json")
        report_path = os.path.join(tmpdir, f"seed_{seed}_report.json")
        argv = [hostpy, os.path.join(REPO_ROOT, "blender_agent", "layout_gen.py"),
                "--seed", str(seed), "--out", path, "--shape", shape, "--quiet",
                "--report", report_path]
        if rooms:
            argv += ["--rooms", str(rooms)]
        result = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT)
        if result.returncode != 0 or not os.path.isfile(path):
            print(f"diversity: seed {seed} failed: {result.stderr.strip()[-200:]}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as handle:
            spec = json.load(handle)
        report = None
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
        out.append((seed, spec, report))
    return out


def _print_summary(summary: dict) -> None:
    print(f"scenes            : {summary.get('count')}")
    print(f"families          : {summary.get('families')}")
    print(f"rooms (histogram) : {summary.get('rooms', {}).get('histogram')}")
    print(f"cycles            : {summary.get('cycles', {}).get('histogram')}")
    print(f"function coverage : {summary.get('function_coverage')}")
    print(f"shape             : {summary.get('shape')}")
    print(f"furniture         : {summary.get('furniture')}")
    print(f"cameras           : {summary.get('cameras')}")
    structures = summary.get("structures", {})
    print(f"topologies        : {structures.get('unique')} unique of {summary.get('count')} "
          f"(duplicate rate {structures.get('duplicate_rate')})")
    for group in structures.get("groups", [])[:5]:
        print(f"  repeat x{group['count']:>2}  {group['hash']}  seeds {group['seeds'][:8]}  "
              f"{group['note']}")
    plans = summary.get("plans", {})
    print(f"plans             : {plans.get('unique')} unique (duplicate rate {plans.get('duplicate_rate')})")


if __name__ == "__main__":
    sys.exit(main())
