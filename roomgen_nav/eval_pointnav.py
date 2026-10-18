#!/usr/bin/env python3
"""PointNav benchmark entry point over the delivered RoomGen scenes.

Run with the Habitat-Sim interpreter::

    LD_LIBRARY_PATH=third_party/nav_sysroot/usr/lib/x86_64-linux-gnu:third_party/nav_env/lib \
      third_party/nav_env/bin/python -m roomgen_nav.eval_pointnav --smoke

The smoke profile is 2 scenes x 5 tasks.  The full run is the 20-scene
inventory in :data:`roomgen_nav.assets.DEFAULT_SCENES`, 50 tasks per scene,
with the same base seed and the same cached NavMeshes / episode files.

Outputs (under ``--out``): ``run.json`` with per-episode results, per-scene and
all-scene summary, the initial states and the stored action lists; ``run.txt``
with a human-readable summary.  Episodes themselves are cached under
``--cache-dir/<scene>/<profile>/episodes_<count>_<seed>.json``.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from roomgen_nav import assets, episodes as episodes_mod, metrics  # noqa: E402
from roomgen_nav.env import PointNavEnv  # noqa: E402
from roomgen_nav.profiles import get_profile  # noqa: E402

SMOKE_SCENES = ("bench_scene_0000", "cross_mezzanine")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="eval_pointnav.py",
                                     description="Run the RoomGen PointNav benchmark")
    parser.add_argument("--scene-ids", default=None,
                        help="comma-separated inventory ids; default: all 20")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50,
                        help="tasks per scene (default 50)")
    parser.add_argument("--profile", default="humanoid",
                        help="humanoid / wheeled / path to a profile JSON")
    parser.add_argument("--seed", type=int, default=20261018,
                        help="base episode seed")
    parser.add_argument("--out", default=None,
                        help="output directory (default under outputs/nav_eval)")
    parser.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, "outputs", "nav_cache"))
    parser.add_argument("--smoke", action="store_true",
                        help="2 scenes x 5 tasks smoke run")
    parser.add_argument("--rebuild-nav", action="store_true",
                        help="ignore cached per-floor NavMeshes")
    parser.add_argument("--rebuild-episodes", action="store_true",
                        help="ignore cached episode catalogs")
    parser.add_argument("--verify-sample", type=int, default=3,
                        help="replay this many stored episodes per scene as a "
                             "reproducibility check (0 = skip; full replay still "
                             "runs every evaluated episode)")
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args(argv)


def select_entries(args) -> List[assets.SceneEntry]:
    if args.smoke:
        entries = [assets.find_scene(scene_id) for scene_id in SMOKE_SCENES]
        args.episodes = 5
    elif args.scene_ids:
        ids = [item.strip() for item in args.scene_ids.split(",") if item.strip()]
        entries = [assets.find_scene(scene_id) for scene_id in ids]
    else:
        entries = assets.default_scenes()
    if args.limit_scenes > 0:
        entries = entries[:args.limit_scenes]
    return entries


def _timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_scene(entry: assets.SceneEntry, profile, args) -> dict:
    from roomgen_nav.navmesh import build_composite_navmesh, require_habitat

    require_habitat()
    started = time.time()
    spec = assets.load_scene(entry)
    source = assets.resolve_mesh(entry)
    nav = build_composite_navmesh(entry, profile, args.cache_dir,
                                  rebuild=args.rebuild_nav, spec=spec)
    catalog, catalog_source = episodes_mod.load_or_generate(
        entry.scene_id, spec, profile, nav, count=args.episodes,
        base_seed=args.seed, cache_root=args.cache_dir,
        rebuild=args.rebuild_episodes)

    verification = {"checked": 0, "failures": [], "ok": True}
    if args.verify_sample > 0:
        verification = episodes_mod.verify_catalog(
            entry.scene_id, spec, profile, nav, catalog,
            limit=min(args.verify_sample, len(catalog)))

    env = PointNavEnv(entry.scene_id, spec, profile, nav)
    results: List[dict] = []
    mismatches = 0
    for episode in catalog.episodes:
        env.reset(entry.scene_id, episode.episode_id, episode.seed, episode=episode)
        env.rollout(episode.actions)
        result = env.evaluate()
        result["reference_success"] = bool(episode.reference_success)
        result["reproducible"] = bool(
            result["success"] == episode.reference_success
            and abs(result["final_distance"] - episode.reference_final_distance) < 1e-6
            and result["steps"] == episode.reference_steps)
        if not result["reproducible"]:
            mismatches += 1
        result["strategy"] = episode.strategy
        result["start_position"] = list(episode.start_position)
        result["start_yaw_deg"] = episode.start_yaw_deg
        result["goal_position"] = list(episode.goal_position)
        result["actions"] = list(episode.actions)
        results.append(result)

    summary = metrics.summarize_results(results)
    record = {
        "scene_id": entry.scene_id,
        "spec_path": entry.spec_path,
        "build_dir": entry.build_dir,
        "source_mesh": source,
        "profile": profile.name,
        "nav": nav.summary(),
        "nav_stats": nav.stats(),
        "catalog_source": catalog_source,
        "catalog": catalog.report,
        "verification": verification,
        "reproducible_episodes": len(results) - mismatches,
        "episode_count": len(results),
        "summary": summary,
        "results": results,
        "seconds": round(time.time() - started, 2),
    }
    return record


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_scenes:
        for entry in assets.default_scenes():
            print(f"{entry.scene_id:34s} spec={entry.spec_path} build={entry.build_dir}")
        return 0
    try:
        from roomgen_nav.navmesh import require_habitat
        require_habitat()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        profile = get_profile(args.profile)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    entries = select_entries(args)
    if not entries:
        print("error: no scenes selected", file=sys.stderr)
        return 2
    out = args.out
    if out is None:
        tag = "smoke" if args.smoke else f"{profile.name}_{args.episodes}"
        out = os.path.join(REPO_ROOT, "outputs", "nav_eval", f"{tag}_seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    run = {
        "kind": "roomgen_pointnav",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "profile": profile.to_dict(),
        "base_seed": args.seed,
        "episodes_per_scene": args.episodes,
        "smoke": bool(args.smoke),
        "scenes": [],
    }

    started = time.time()
    for number, entry in enumerate(entries, 1):
        print(f"[{number}/{len(entries)}] {entry.scene_id}: building nav + episodes ...",
              flush=True)
        try:
            record = _run_scene(entry, profile, args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            run["scenes"].append({"scene_id": entry.scene_id, "error": str(exc),
                                  "results": [], "summary": metrics.summarize_results([])})
            continue
        run["scenes"].append(record)
        summary = record["summary"]
        print(f"    floors={record['nav_stats']['floors']} "
              f"links={record['nav_stats']['valid_links']}/{record['nav_stats']['links']} "
              f"episodes={record['episode_count']} "
              f"success={summary['success_rate']:.3f} SPL={summary['spl']:.3f} "
              f"collisions={summary['collisions']} "
              f"reproducible={record['reproducible_episodes']}/{record['episode_count']} "
              f"({record['seconds']}s)", flush=True)
        if record["catalog"]["generated"] < record["catalog"]["requested"]:
            print(f"    warning: only {record['catalog']['generated']} of "
                  f"{record['catalog']['requested']} episodes generated", flush=True)
        gc.collect()

    metrics.aggregate_run(run)
    run["seconds"] = round(time.time() - started, 2)
    run_path = os.path.join(out, "run.json")
    metrics.write_run(run_path, run)
    summary_path = os.path.join(out, "run.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"RoomGen PointNav  profile={profile.name}  scenes={len(run['scenes'])}\n")
        handle.write("=" * 72 + "\n")
        handle.write(metrics.format_summary(run["summary"]) + "\n\n")
        for scene in run["scenes"]:
            if scene.get("error"):
                handle.write(f"{scene['scene_id']}: ERROR {scene['error']}\n")
                continue
            scene_summary = scene.get("summary", {})
            handle.write(
                f"{scene['scene_id']:34s} success={scene_summary.get('success_rate', 0.0):.3f} "
                f"SPL={scene_summary.get('spl', 0.0):.3f} "
                f"coll={scene_summary.get('collisions', 0):4d} "
                f"timeout={scene_summary.get('timeout_rate', 0.0):.3f} "
                f"fd={scene_summary.get('final_distance_mean', 0.0):.3f}\n")
    print("\n" + metrics.format_summary(run["summary"]))
    print(f"\nrun written: {run_path}")
    print(f"summary    : {summary_path}")
    if run["summary"]["episodes"] == 0:
        print("error: no episodes were evaluated", file=sys.stderr)
        return 2
    if args.fail_on_mismatch:
        mismatches = sum(scene.get("episode_count", 0) - scene.get("reproducible_episodes", 0)
                         for scene in run["scenes"] if not scene.get("error"))
        if mismatches:
            print(f"error: {mismatches} stored actions did not reproduce", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
