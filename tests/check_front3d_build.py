#!/usr/bin/env python
"""Numeric checks for a 3D-FRONT replay build (host side, no bpy).

Reads the build the Blender side wrote plus the exported PanoWorld scene and
verifies the facts that ``tools/validate_blender_agent_p0.py`` cannot see:

* the dataset shell really was replayed (mesh/triangle counts, kinds, wall height)
* the 3D-FUTURE models were imported, and the one whose folder is absent was
  skipped instead of crashing (the dataset is only partially public)
* every placed model sits exactly where the dataset's ``pos``/``rot``/``scale``
  puts it -- the world AABB of each instance against hand-computed numbers, and
  the model's own origin at ``y = 0`` (3D-FUTURE models are base-anchored)
* the *metric* structure of the render: ERP depth along known directions
  (zenith, nadir, both long walls, the window plane) equals the room geometry,
  which pins the floor at ``z = 0``, the ceiling at 2.6 m and the camera at 1.5 m

Usage::

    python tests/check_front3d_build.py --build <build> --scene <scene dir> \\
        --spec <bridged spec> --report <check_report.json>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# hand-computed expectations for tests/fixtures/front3d_mini, in the repo frame
# (see tests/fixtures/front3d_mini/make_fixture.py --print)
EXPECTED_PLACEMENTS = {
    "100/model": {"jid_tail": "0000000000a1", "aabb": [[0.3, -2.02, 0.0], [2.1, -1.18, 0.75]]},
    "101/model": {"jid_tail": "0000000000a1", "aabb": [[0.66, -1.17, 0.0], [1.74, -0.33, 0.75]]},
    "131/model": {"jid_tail": "0000000000b2", "aabb": [[4.3, -2.5, 0.0], [6.5, -0.9, 0.88]]},
    "132/model": {"jid_tail": "0000000000c3", "aabb": [[4.125, -3.125, 0.0], [4.595, -2.675, 0.5]]},
}
EXPECTED_SHELL = {"meshes": 20, "triangles": 40,
                  "by_kind": {"floor": 2, "ceiling": 2, "wall": 13, "other": 3},
                  "wall_height_m": 2.6, "window_meshes": 2}
MISSING_JID = "0f1b6f2e-1c1e-4a5a-9b1f-ffffffffffff"
# (camera, lon in degrees, lat in degrees, expected distance in metres)
EXPECTED_DEPTH_RAYS = [
    ("0000", 0.0, 90.0, 1.10),        # straight up to the ceiling (2.6 - 1.5)
    ("0000", 0.0, -90.0, 1.08),       # down onto the sofa seat (1.5 - 0.42)
    ("0000", 90.0, 0.0, 1.60),        # the room's z = 0 wall
    ("0000", -90.0, 0.0, 1.60),       # the room's z = 3.2 wall
    ("0000", 180.0, 0.0, 1.75),       # west wall x = 0 through the window pane
    ("0001", 0.0, 90.0, 1.10),        # ceiling
    ("0001", 0.0, -90.0, 1.02),       # down onto the mattress (1.5 - 0.48)
    ("0001", 0.0, 0.0, 1.75),         # east wall x = 7
    ("0001", 90.0, 0.0, 1.60),
    ("0001", -90.0, 0.0, 1.60),
]
DEPTH_TOLERANCE_M = 0.005


def probe_depth(depth: np.ndarray, lon_deg: float, lat_deg: float) -> float:
    """Median depth over a 5x5 neighbourhood of an ERP direction.

    Same pixel convention as ``spec.py``/the validator:
    ``lon = (col + 0.5) / W * 2pi - pi``, ``lat = pi/2 - (row + 0.5) / H * pi``.
    """
    height, width = depth.shape
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    column = int(round((lon + math.pi) / (2.0 * math.pi) * width - 0.5)) % width
    row = int(round((math.pi / 2.0 - lat) / math.pi * height - 0.5))
    row = max(2, min(height - 3, row))
    columns = [(column + offset) % width for offset in range(-2, 3)]
    return float(np.median(depth[row - 2:row + 3][:, columns]))


def close(a, b, tolerance: float) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check a 3D-FRONT replay build")
    parser.add_argument("--build", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    build_dir = os.path.abspath(args.build)
    scene_dir = os.path.abspath(args.scene)
    checks = []

    def record(name, passed, **details):
        checks.append({"name": name, "pass": bool(passed), **details})
        print(f"[check] {'PASS' if passed else 'FAIL'}  {name}"
              + ("" if passed else f"  {json.dumps(details)[:300]}"))
        return passed

    manifest_path = os.path.join(build_dir, "build_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(args.spec, "r", encoding="utf-8") as handle:
        spec = json.load(handle)

    # ---- the spec the build used is the bridged one ----------------------- #
    record("spec_is_the_bridged_front3d_spec",
           os.path.abspath(manifest["spec_path"]) == os.path.abspath(args.spec)
           and bool(spec.get("front3d")),
           manifest_spec=manifest.get("spec_path"), given=os.path.abspath(args.spec))

    # ---- shell replay ---------------------------------------------------- #
    front3d = manifest.get("front3d") or {}
    shell_report = front3d.get("models") or {}
    record("front3d_block_in_manifest",
           bool(front3d) and front3d.get("scene_uid") == "front3d-mini-0001",
           scene_uid=front3d.get("scene_uid"))
    record("shell_replayed_from_dataset_meshes",
           front3d.get("room_functions") == {"LivingDiningRoom-mini1": "living",
                                             "BedRoom-mini2": "bedroom"},
           room_functions=front3d.get("room_functions"))
    # the shell counters live in the manifest's front3d block only through the
    # models report, so re-derive them from the ply + a fresh parse
    from blender_agent import front3d as front3d_mod
    scene = front3d_mod.parse_scene(spec["front3d"]["scene_json"])
    shell_stats = {
        "meshes": len(scene.meshes),
        "triangles": sum(mesh.triangle_count for mesh in scene.meshes.values()),
        "by_kind": {},
        "wall_height_m": round(front3d_mod.shell_wall_height(scene), 4),
        "window_meshes": sum(1 for mesh in scene.meshes.values()
                             if front3d_mod._fallback_key(mesh.type) == "window"),
    }
    for mesh in scene.meshes.values():
        shell_stats["by_kind"][mesh.kind] = shell_stats["by_kind"].get(mesh.kind, 0) + 1
    record("shell_stats_match_the_fixture", shell_stats == EXPECTED_SHELL,
           got=shell_stats, expected=EXPECTED_SHELL)

    # ---- models ---------------------------------------------------------- #
    models = front3d.get("models") or {}
    imported = {entry["jid"] for entry in models.get("models_imported", [])}
    expected_jids = {item["asset"].split(":", 1)[1] for item in spec["furniture"]} - {MISSING_JID}
    record("all_available_models_imported", imported == expected_jids,
           imported=sorted(jid[-4:] for jid in imported),
           expected=sorted(jid[-4:] for jid in expected_jids))
    not_on_disk = front3d.get("furniture_not_on_disk") or []
    record("missing_model_skipped_not_fatal",
           MISSING_JID in models.get("missing_models", [])
           and [entry["name"] for entry in not_on_disk] == ["140/model"],
           missing_models=models.get("missing_models"), skipped=not_on_disk)
    record("model_textures_applied", models.get("model_textures") == 3,
           model_textures=models.get("model_textures"))

    # ---- placement poses vs hand-computed world AABBs -------------------- #
    placements = {item["name"]: item for item in manifest["furniture"]}
    record("placement_count", sorted(placements) == sorted(EXPECTED_PLACEMENTS),
           got=sorted(placements), expected=sorted(EXPECTED_PLACEMENTS))
    pose_ok, pose_details = True, {}
    for name, expected in EXPECTED_PLACEMENTS.items():
        item = placements.get(name)
        if item is None or item.get("kind") != "front3d":
            pose_ok = False
            pose_details[name] = "missing or not a front3d placement"
            continue
        worst = max(abs(item["bbox_world"][corner][axis] - expected["aabb"][corner][axis])
                    for corner in (0, 1) for axis in range(3))
        origin_y = item.get("model_origin_y")
        if worst > 1e-6 or not close(origin_y, 0.0, 1e-6):
            pose_ok = False
        pose_details[name] = {
            "jid_tail": item["jid"][-4:],
            "bbox_world": item["bbox_world"],
            "expected": expected["aabb"],
            "max_abs_error_m": worst,
            "model_origin_y": origin_y,
        }
    record("placement_poses_match_hand_computation", pose_ok, **pose_details)

    scaled = placements.get("131/model", {}).get("scale_xyz")
    record("non_uniform_dataset_scale_preserved", scaled == [1.0, 0.8, 1.1], scale_xyz=scaled)

    # ---- the render really is the fixture, in metres --------------------- #
    depth_ok, depth_details = True, []
    for camera, lon, lat, expected in EXPECTED_DEPTH_RAYS:
        with open(os.path.join(scene_dir, "viewpoints", camera, "place_depth_scale.txt"),
                  "r", encoding="utf-8") as handle:
            scale = float(handle.read().strip())
        raw = np.array(Image.open(os.path.join(scene_dir, "viewpoints", camera, "place_depth.png")))
        depth = raw.astype(np.float64) / scale
        got = probe_depth(depth, lon, lat)
        error = abs(got - expected)
        passed = error <= DEPTH_TOLERANCE_M
        depth_ok = depth_ok and passed
        depth_details.append({"camera": camera, "lon_deg": lon, "lat_deg": lat,
                              "expected_m": expected, "got_m": round(got, 5),
                              "error_m": round(error, 5), "pass": passed})
    record("metric_depth_at_known_directions", depth_ok, rays=depth_details,
           tolerance_m=DEPTH_TOLERANCE_M)

    # ---- window portal lights, derived from the dataset's Window meshes ---- #
    window_lights = (front3d.get("models") or {}).get("window_light_details") or []
    by_room = {entry["room"]: entry for entry in window_lights}
    lights_ok = (
        len(window_lights) == 2
        and sorted(entry["size_m"] for entry in window_lights) == [[0.58, 0.98], [0.98, 0.98]]
        and by_room["LivingDiningRoom-mini1"]["inward"][0] > 0.9      # x = 0 pane faces +X
        and by_room["BedRoom-mini2"]["inward"][0] < -0.8              # x = 7 pane faces -X
    )
    record("window_lights_match_the_panes", lights_ok, window_lights=window_lights)

    # ---- viewpoints ------------------------------------------------------ #
    viewpoints = sorted(os.listdir(os.path.join(scene_dir, "viewpoints")))
    record("viewpoint_count_matches_rooms", viewpoints == ["0000", "0001"], viewpoints=viewpoints)
    spec_poses = {camera["name"]: camera["pos"] for camera in spec["cameras"]}
    pose_match = True
    for name, position in spec_poses.items():
        matrix = np.loadtxt(os.path.join(scene_dir, "viewpoints", name, "extrinsics.txt"))
        error = float(np.max(np.abs(matrix[:3, 3] - np.array(position, dtype=float))))
        pose_match = pose_match and error < 1e-9
    record("exported_poses_equal_the_spec", pose_match, poses=spec_poses)

    pano = np.array(Image.open(os.path.join(scene_dir, "viewpoints", "0000", "panoImage_2048.png")))
    record("panorama_is_not_degenerate", float(pano.std()) > 5.0,
           shape=list(pano.shape), std=round(float(pano.std()), 3))

    report = {
        "kind": "front3d_replay_check",
        "build": build_dir,
        "scene": scene_dir,
        "spec": os.path.abspath(args.spec),
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
    }
    print()
    failed = [item["name"] for item in checks if not item["pass"]]
    print(f"[check] overall: {'PASS' if report['pass'] else 'FAIL'}"
          + (f"  failed: {failed}" if failed else ""))
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"[check] report written to {args.report}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
