#!/usr/bin/env python3
"""PointNav benchmark entry point over the delivered RoomGen scenes.

Run with the Habitat-Sim interpreter::

    LD_LIBRARY_PATH=third_party/nav_sysroot/usr/lib/x86_64-linux-gnu:third_party/nav_env/lib \
      third_party/nav_env/bin/python -m roomgen_nav.eval_pointnav --smoke

Policies
--------
``--policy path`` (default)
    Closed-loop :class:`roomgen_nav.env.ShortestPathPolicy`: every step reads
    the current position / heading and the episode's ``reference_path``
    waypoints, then executes ``left`` / ``right`` / ``forward`` / ``stop``.
    This is the policy result.

``--policy oracle``
    Replays the stored reference action list.  This is a path executability
    sanity check, not a policy; a 100% oracle rate says the reference paths can
    be walked, not that an agent can find them.

In ``path`` mode the oracle replay also runs for every episode and is reported
separately (``run["oracle"]`` / ``run.txt``), so the two numbers are never
conflated.

Outputs (under ``--out``): ``run.json`` with per-episode policy results, the
separate policy / oracle summaries and per-scene records; ``run.txt`` with the
two summaries side by side; ``episodes.jsonl`` with one step-by-step rollout per
line (the ``tools/build_nav_replay.py`` contract).  Episodes are cached under
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
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from roomgen_nav import assets, episodes as episodes_mod, metrics  # noqa: E402
from roomgen_nav.env import (  # noqa: E402
    PointNavEnv,
    ShortestPathPolicy,
    run_action_sequence,
)
from roomgen_nav.profiles import get_profile  # noqa: E402

SMOKE_SCENES = ("bench_scene_0000", "cross_mezzanine")

ORACLE_KEYS = ("success", "spl", "collisions", "steps", "timeout",
               "final_distance", "failure_reason")


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
    parser.add_argument("--policy", choices=("path", "oracle"), default="path",
                        help="path: closed-loop waypoint policy (default); "
                             "oracle: stored action replay sanity check")
    parser.add_argument("--seed", type=int, default=20261018,
                        help="base episode seed")
    parser.add_argument("--out", default=None,
                        help="output directory (default under outputs/nav_eval)")
    parser.add_argument("--episodes-jsonl", default=None,
                        help="step-by-step rollout file (default <out>/episodes.jsonl)")
    parser.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, "outputs", "nav_cache"))
    parser.add_argument("--smoke", action="store_true",
                        help="2 scenes x 5 tasks smoke run")
    parser.add_argument("--rebuild-nav", action="store_true",
                        help="ignore cached per-floor NavMeshes")
    parser.add_argument("--rebuild-episodes", action="store_true",
                        help="ignore cached episode catalogs")
    parser.add_argument("--verify-sample", type=int, default=3,
                        help="extra stored-action replays per scene "
                             "(0 = skip; path mode already replays every episode)")
    parser.add_argument("--repro-check", action="store_true",
                        help="run the path policy twice per episode and compare "
                             "the whole action sequence")
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


def _oracle_matches_reference(result: dict, episode) -> bool:
    return bool(
        result["success"] == episode.reference_success
        and abs(result["final_distance"] - episode.reference_final_distance) < 1e-6
        and result["steps"] == episode.reference_steps)


def _runs_match(first: dict, second: dict) -> bool:
    if first["actions"] != second["actions"]:
        return False
    left = [step["position"] for step in first["trajectory"]]
    right = [step["position"] for step in second["trajectory"]]
    return left == right


def _run_scene(entry: assets.SceneEntry, profile, args, episodes_handle=None) -> dict:
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
    # The policy must observe the arrival and issue ``stop`` itself; the oracle
    # replay keeps the historical immediate-success behaviour.
    policy_env = (PointNavEnv(entry.scene_id, spec, profile, nav, auto_success=False)
                  if args.policy == "path" else env)
    policy = ShortestPathPolicy(policy_env)
    results: List[dict] = []
    oracle_results: List[dict] = []
    oracle_mismatches = 0
    policy_mismatches = 0

    for episode in catalog.episodes:
        env.reset(entry.scene_id, episode.episode_id, episode.seed, episode=episode)
        oracle_run = run_action_sequence(env, episode.actions)
        oracle_result = oracle_run["result"]
        oracle_reproducible = _oracle_matches_reference(oracle_result, episode)
        if not oracle_reproducible:
            oracle_mismatches += 1
        oracle_result["policy"] = "oracle"
        oracle_result["reference_success"] = bool(episode.reference_success)
        oracle_result["reproducible"] = bool(oracle_reproducible)
        oracle_result["strategy"] = episode.strategy
        oracle_result["start_position"] = list(episode.start_position)
        oracle_result["start_yaw_deg"] = episode.start_yaw_deg
        oracle_result["goal_position"] = list(episode.goal_position)
        oracle_result["actions"] = list(oracle_run["actions"])
        oracle_results.append(oracle_result)

        if args.policy == "path":
            policy_run = policy.run(episode)
            result = policy_run["result"]
            actions = policy_run["actions"]
            trajectory = policy_run["trajectory"]
            reproducible = True
            if args.repro_check:
                repeated = policy.run(episode)
                reproducible = _runs_match(policy_run, repeated)
                if not reproducible:
                    policy_mismatches += 1
            result["policy"] = "path"
            result["reproducible"] = bool(reproducible)
            result["oracle"] = {key: oracle_result[key] for key in ORACLE_KEYS}
        else:  # oracle-only run: the replay is the evaluated result
            result = oracle_result
            actions = oracle_run["actions"]
            trajectory = oracle_run["trajectory"]
            result["policy"] = "oracle"
            result["reproducible"] = bool(oracle_reproducible)

        result["reference_success"] = bool(episode.reference_success)
        result["strategy"] = episode.strategy
        result["start_position"] = list(episode.start_position)
        result["start_yaw_deg"] = episode.start_yaw_deg
        result["goal_position"] = list(episode.goal_position)
        result["actions"] = list(actions)
        results.append(result)

        if episodes_handle is not None:
            episodes_mod.write_episode_jsonl(
                episodes_handle,
                episodes_mod.episode_record(
                    episode, result, trajectory, policy=args.policy,
                    profile=profile))

    summary = metrics.summarize_results(results)
    oracle_summary = metrics.summarize_results(oracle_results)
    record = {
        "scene_id": entry.scene_id,
        "spec_path": entry.spec_path,
        "build_dir": entry.build_dir,
        "source_mesh": source,
        "replay_scene": _scene_geometry(entry, spec),
        "profile": profile.name,
        "policy": args.policy,
        "policy_params": policy.describe() if args.policy == "path" else None,
        "nav": nav.summary(),
        "nav_stats": nav.stats(),
        "catalog_source": catalog_source,
        "catalog": catalog.report,
        "verification": verification,
        "oracle_reproducible_episodes": len(oracle_results) - oracle_mismatches,
        "policy_reproducible_episodes": (
            len(results) - policy_mismatches if args.policy == "path"
            and args.repro_check else None),
        "episode_count": len(results),
        "summary": summary,
        "oracle_summary": oracle_summary,
        "results": results,
        "oracle_results": oracle_results,
        "seconds": round(time.time() - started, 2),
    }
    return record


def _polygon_gltf(points) -> List[List[float]]:
    """Blender XY (spec) -> the replay tool's glTF XZ floor plane."""
    return [[float(point[0]), -float(point[1])] for point in points or []]


def _scene_geometry(entry: assets.SceneEntry, spec) -> dict:
    """Scene polygons in the frame ``build_nav_replay.py`` expects.

    Episode positions are Habitat Y-up ``(x, y, z)`` with ``z = -spec_y``, which
    is the same plane as glTF ``(x, z)``.  Rooms, slab openings, voids and doors
    are therefore converted once here instead of letting the replay tool guess.
    """
    floors = []
    if spec.floors:
        for floor in spec.floors:
            floors.append({"id": floor.id, "name": floor.id,
                           "elevation": float(floor.elevation),
                           "height": float(floor.height)})
    else:
        floors.append({"id": "F0", "name": "F0", "elevation": 0.0,
                       "height": float(spec.wall_height)})
    elevation_by_floor = {floor["id"]: floor["elevation"] for floor in floors}

    rooms = []
    for room in spec.rooms:
        floor_id = assets.floor_id_of_room(spec, room.id)
        rooms.append({
            "id": room.id,
            "type": room.type,
            "floor_id": floor_id,
            "polygon": _polygon_gltf(room.polygon),
            "elevation": elevation_by_floor.get(floor_id, 0.0),
        })

    voids = []
    for index, opening in enumerate(spec.slab_openings):
        for floor_id in opening.floors:
            voids.append({"id": f"slab_opening_{index}_{floor_id}",
                          "floor_id": floor_id,
                          "polygon": _polygon_gltf(opening.outline),
                          "label": "slab_opening"})
    for index, void in enumerate(spec.voids):
        polygon = _polygon_gltf(void.outline)
        if spec.floors:
            try:
                lower = spec.floor(void.floor).elevation
            except Exception:
                lower = float("-inf")
            for floor in spec.floors:
                if floor.elevation > lower + 1e-9:
                    voids.append({"id": f"void_{index}_{floor.id}",
                                  "floor_id": floor.id, "polygon": polygon,
                                  "label": "void"})
        else:
            voids.append({"id": f"void_{index}", "floor_id": None,
                          "polygon": polygon, "label": "void"})

    portals = []
    for index, connection in enumerate(spec.door_connections()):
        center = connection.center
        if not center or len(center) < 2:
            continue
        floor_id = connection.from_floor or None
        if connection.from_floor and connection.to_floor \
                and connection.from_floor != connection.to_floor:
            floor_id = None
        portals.append({
            "id": f"door_{index}",
            "floor_id": floor_id,
            "position": [float(center[0]), -float(center[1])],
            "radius": max(0.3, float(connection.width) / 2.0),
            "label": connection.kind,
        })

    return {
        "scene_id": entry.scene_id,
        "coordinate_system": "gltf",
        "floors": floors,
        "rooms": rooms,
        "voids": voids,
        "portals": portals,
    }


def _primary_heading(policy_name: str) -> str:
    if policy_name == "path":
        return "=== POLICY: path (closed-loop result) ==="
    return ("=== PRIMARY: oracle (stored action replay; executability "
            "sanity check, NOT a policy) ===")


def _write_run_txt(path: str, run: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f"RoomGen PointNav  profile={run['profile']['name']}  "
            f"scenes={len(run['scenes'])}  policy={run['policy_name']}\n")
        handle.write("=" * 72 + "\n")
        handle.write(metrics.format_summary(
            run["policy"]["summary"],
            heading=_primary_heading(run["policy_name"])) + "\n\n")
        handle.write(metrics.format_summary(
            run["oracle"]["summary"],
            heading="=== ORACLE: stored action replay "
                    "(path executability sanity check, NOT a policy) ===") + "\n\n")
        for scene in run["scenes"]:
            if scene.get("error"):
                handle.write(f"{scene['scene_id']}: ERROR {scene['error']}\n")
                continue
            scene_summary = scene.get("summary", {})
            oracle_summary = scene.get("oracle_summary", {})
            handle.write(
                f"{scene['scene_id']:34s} "
                f"policy success={scene_summary.get('success_rate', 0.0):.3f} "
                f"SPL={scene_summary.get('spl_all', 0.0):.3f} "
                f"coll={scene_summary.get('collisions', 0):4d} "
                f"timeout={scene_summary.get('timeout_rate', 0.0):.3f} "
                f"fd={scene_summary.get('final_distance_mean', 0.0):.3f} "
                f"| oracle success={oracle_summary.get('success_rate', 0.0):.3f} "
                f"SPL={oracle_summary.get('spl_all', 0.0):.3f}\n")


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
        if args.smoke:
            tag = f"smoke_{args.policy}"
        else:
            tag = f"{profile.name}_{args.episodes}_{args.policy}"
        out = os.path.join(REPO_ROOT, "outputs", "nav_eval", f"{tag}_seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    episodes_jsonl = args.episodes_jsonl or os.path.join(out, "episodes.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(episodes_jsonl)), exist_ok=True)

    run = {
        "kind": "roomgen_pointnav",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "policy_name": args.policy,
        "policy_note": (
            "path: closed-loop ShortestPathPolicy over reference_path waypoints; "
            "oracle: stored reference action replay (executability sanity check)"
        ),
        "repro_check": bool(args.repro_check),
        "profile": profile.to_dict(),
        "base_seed": args.seed,
        "episodes_per_scene": args.episodes,
        "smoke": bool(args.smoke),
        "episodes_jsonl": os.path.abspath(episodes_jsonl),
        "scenes": [],
    }

    started = time.time()
    with open(episodes_jsonl, "w", encoding="utf-8") as episodes_handle:
        for number, entry in enumerate(entries, 1):
            print(f"[{number}/{len(entries)}] {entry.scene_id}: "
                  f"building nav + episodes ...", flush=True)
            try:
                record = _run_scene(entry, profile, args,
                                    episodes_handle=episodes_handle)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                run["scenes"].append({
                    "scene_id": entry.scene_id, "error": str(exc),
                    "results": [], "oracle_results": [],
                    "summary": metrics.summarize_results([]),
                    "oracle_summary": metrics.summarize_results([])})
                continue
            run["scenes"].append(record)
            summary = record["summary"]
            oracle_summary = record["oracle_summary"]
            print(f"    floors={record['nav_stats']['floors']} "
                  f"links={record['nav_stats']['valid_links']}/"
                  f"{record['nav_stats']['links']} "
                  f"episodes={record['episode_count']}\n"
                  f"    policy: success={summary['success_rate']:.3f} "
                  f"SPL(all)={summary['spl_all']:.3f} "
                  f"collisions={summary['collisions']} "
                  f"timeouts={summary['timeouts']} "
                  f"fd={summary['final_distance_mean']:.3f}\n"
                  f"    oracle: success={oracle_summary['success_rate']:.3f} "
                  f"SPL(all)={oracle_summary['spl_all']:.3f} "
                  f"({record['seconds']}s)", flush=True)
            if record["catalog"]["generated"] < record["catalog"]["requested"]:
                print(f"    warning: only {record['catalog']['generated']} of "
                      f"{record['catalog']['requested']} episodes generated",
                      flush=True)
            if args.repro_check and record["policy_reproducible_episodes"] is not None:
                print(f"    policy reproducibility: "
                      f"{record['policy_reproducible_episodes']}/"
                      f"{record['episode_count']}", flush=True)
            gc.collect()

    metrics.aggregate_run(run)
    all_oracle = [result for scene in run["scenes"]
                  for result in scene.get("oracle_results", [])]
    oracle_summary = metrics.summarize_results(all_oracle)
    policy_report = {
        "name": args.policy,
        "summary": run["summary"],
        "by_scene": run["scene_summaries"],
        "note": ("closed-loop ShortestPathPolicy over reference waypoints"
                 if args.policy == "path"
                 else "stored action replay (legacy oracle mode; not a policy)"),
    }
    oracle_report = {
        "name": "oracle",
        "summary": oracle_summary,
        "by_scene": metrics.summarize_by_scene({
            scene["scene_id"]: scene.get("oracle_results", [])
            for scene in run["scenes"] if not scene.get("error")}),
        "note": ("stored reference action replay: path executability sanity "
                 "check; not a policy success rate"),
    }
    run["policy"] = policy_report
    run["oracle"] = oracle_report
    run["summary_scope"] = "policy"
    run["seconds"] = round(time.time() - started, 2)
    scenes_path = os.path.join(out, "scenes.json")
    with open(scenes_path, "w", encoding="utf-8") as handle:
        json.dump({
            "kind": "roomgen_nav_scenes",
            "coordinate_system": "gltf",
            "scenes": [scene["replay_scene"] for scene in run["scenes"]
                       if not scene.get("error") and scene.get("replay_scene")],
        }, handle, indent=2)
    run["scenes_json"] = os.path.abspath(scenes_path)
    run_path = os.path.join(out, "run.json")
    metrics.write_run(run_path, run)
    summary_path = os.path.join(out, "run.txt")
    _write_run_txt(summary_path, run)
    print("\n" + metrics.format_summary(
        policy_report["summary"], heading=_primary_heading(args.policy)))
    print("\n" + metrics.format_summary(
        oracle_report["summary"],
        heading="=== ORACLE: stored action replay (sanity check, NOT a policy) ==="))
    print(f"\nrun written : {run_path}")
    print(f"summary     : {summary_path}")
    print(f"episodes    : {episodes_jsonl}")
    print(f"scenes      : {scenes_path}")
    if run["summary"]["episodes"] == 0:
        print("error: no episodes were evaluated", file=sys.stderr)
        return 2
    if args.fail_on_mismatch:
        oracle_mismatches = sum(
            scene.get("episode_count", 0) - scene.get("oracle_reproducible_episodes", 0)
            for scene in run["scenes"] if not scene.get("error"))
        policy_mismatches = 0
        if args.repro_check:
            for scene in run["scenes"]:
                if scene.get("error"):
                    continue
                reproducible = scene.get("policy_reproducible_episodes")
                if reproducible is None:
                    reproducible = scene.get("episode_count", 0)
                policy_mismatches += scene.get("episode_count", 0) - reproducible
        if oracle_mismatches or policy_mismatches:
            print(f"error: {oracle_mismatches} oracle replays and "
                  f"{policy_mismatches} policy reruns did not reproduce",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
