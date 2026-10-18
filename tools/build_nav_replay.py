#!/usr/bin/env python3
"""Build a self-contained, browser-only replay site for PointNav episodes.

The evaluator writes one JSON object per line (``episodes.jsonl``) with the
fields

    scene_id, episode_id, seed, start, goal, start_floor, goal_floor,
    reference_path {points, distance}, trajectory [{position, yaw, action,
    collision}], success, spl, collisions, timeout, failure_reason

and (optionally) sibling ``scenes.json`` / ``metrics.json`` files.  This tool
normalises those inputs, attaches floor polygons from a scene manifest when one
is available, classifies failures (stuck at a door, through-wall, cross-void,
wrong-floor jump, ...), and writes a static site that needs no network access:

    <out>/index.html
    <out>/replay.css
    <out>/replay.js
    <out>/replay-data.js       window.NAV_REPLAY_DATA = {...}
    <out>/replay_manifest.json small machine-readable build report

The page is plain HTML + Canvas 2D.  It never fetches anything, so it also
works when ``index.html`` is opened directly from disk.

Usage::

    python tools/build_nav_replay.py \\
        --episodes outputs/nav_eval/run01/episodes.jsonl \\
        --scenes   outputs/nav_eval/run01/scenes.json \\
        --metrics  outputs/nav_eval/run01/metrics.json \\
        --out      outputs/nav_eval/run01/replay

    # or let the tool discover the sibling files
    python tools/build_nav_replay.py --episodes outputs/nav_eval/run01 --out ...

    # local preview (no dependency beyond the standard library)
    python tools/build_nav_replay.py --episodes ... --out ... --serve --port 8765
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "nav_replay"
ASSET_FILES = ("index.html", "replay.css", "replay.js")
EPISODE_FILENAME = "episodes.jsonl"

CATEGORY_LABELS = {
    "success": "Success",
    "stuck_door": "Stuck at door",
    "stuck": "Stuck / blocked",
    "through_wall": "Through wall",
    "cross_void": "Crossed void / hole",
    "floor_jump": "Wrong floor jump",
    "collision": "Collision",
    "timeout": "Timeout",
    "other": "Other failure",
}

REASON_KEYWORDS = (
    ("cross_void", (
        "cross_void", "cross void", "void crossing", "fell through",
        "fall through", "dropped through", "drop through", "fell", "hole",
        "void", "gap", "跨洞", "坠落", "掉入", "掉下", "洞里",
    )),
    ("through_wall", (
        "through_wall", "through wall", "wall penetration", "penetrated",
        "penetration", "穿墙", "穿墙而过",
    )),
    ("floor_jump", (
        "floor_jump", "floor jump", "wrong_floor", "wrong floor",
        "wrong level", "wrong layer", "wrong storey", "wrong story",
        "jumped floor", "layer jump", "跳层", "错层", "跨层",
    )),
    ("stuck_door", (
        "stuck_door", "stuck at door", "stuck in door", "doorway",
        "blocked by door", "卡门", "卡在门", "门框",
    )),
    ("stuck", ("stuck", "blocked", "卡住", "卡死")),
    ("collision", ("collision", "collided", "碰撞", "撞到")),
    ("timeout", ("timeout", "timed out", "time limit", "超时")),
)

SPECIFIC_CATEGORIES = ("cross_void", "through_wall", "floor_jump", "stuck_door")

DEFAULT_SUPPORT_TOLERANCE = 0.3  # metres inside/outside a room polygon


# --------------------------------------------------------------------------- #
# small parsing helpers
# --------------------------------------------------------------------------- #
def as_float(value):
    """Return a finite float or ``None`` (booleans are not numbers here)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rounded(value, digits=4):
    number = as_float(value)
    return None if number is None else round(number, digits)


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "yes", "1", "success", "succeeded"):
            return True
        if text in ("false", "no", "0", "failure", "failed", ""):
            return False
    return None


def as_int(value):
    number = as_float(value)
    return None if number is None else int(round(number))


def position3(value, fallback_y=0.0):
    """Normalise a point to ``[x, y, z]``.

    Accepted shapes: ``[x, y, z]``, ``[x, z]`` (floor plane), ``{x, y, z}``,
    ``{x, z}``, ``{x, y}`` (floor plane) and wrappers such as
    ``{"position": ...}``.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        numbers = [as_float(item) for item in value]
        if len(numbers) >= 3:
            x, y, z = numbers[0], numbers[1], numbers[2]
            if x is None or y is None or z is None:
                return None
            return [x, y, z]
        if len(numbers) == 2:
            x, z = numbers
            if x is None or z is None:
                return None
            return [x, fallback_y, z]
        return None
    if not isinstance(value, dict):
        return None
    for key in (
        "position", "point", "xyz", "center", "translation",
        "location", "world_position",
    ):
        if key in value:
            point = position3(value[key], fallback_y=fallback_y)
            if point is not None:
                return point
    x = as_float(value.get("x"))
    y = as_float(value.get("y"))
    z = as_float(value.get("z"))
    if x is not None and z is not None:
        return [x, fallback_y if y is None else y, z]
    if x is not None and y is not None and "z" not in value:
        # ``{x, y}`` is a floor-plane point, so the second number is z.
        return [x, fallback_y, y]
    return None


def polygon2(value):
    """Return a footprint as ``[[x, z], ...]``."""
    if not isinstance(value, (list, tuple)):
        return []
    points = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x, z = as_float(item[0]), as_float(item[1])
        elif isinstance(item, dict):
            x = as_float(item.get("x"))
            z = as_float(item.get("z"))
            if z is None:
                z = as_float(item.get("y"))
        else:
            continue
        if x is not None and z is not None:
            points.append([x, z])
    return points


def coord_mode(value):
    text = str(value or "gltf").strip().lower()
    if text in ("blender", "z_up", "zup", "blender_z_up"):
        return "blender"
    return "gltf"


def yaw_unit_name(value):
    text = str(value or "").strip().lower()
    if text in ("degree", "degrees", "deg", "d"):
        return "degrees"
    if text in ("radian", "radians", "rad"):
        return "radians"
    return None


def yaw_convention_name(value):
    text = str(value or "").strip().lower()
    if text in ("manifest", "roomgen", "env", "x_forward", "gltf"):
        return "manifest"
    if text in ("habitat", "habitat_sim", "minus_z"):
        return "habitat"
    return None


def blender_position_to_gltf(point):
    x, y, z = point
    return [x, z, -y]


def blender_polygon_to_gltf(points):
    return [[x, -y] for x, y in points]


def point_in_polygon(x, z, polygon, tolerance=0.0):
    """Ray-cast point-in-polygon with an optional boundary tolerance."""
    if len(polygon) < 3:
        return False
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, z1 = polygon[index]
        x2, z2 = polygon[(index + 1) % count]
        if _point_segment_distance(x, z, x1, z1, x2, z2) <= tolerance:
            return True
        if (z1 > z) != (z2 > z):
            cross_x = x1 + (z - z1) * (x2 - x1) / (z2 - z1)
            if cross_x > x:
                inside = not inside
    return inside


def _point_segment_distance(px, pz, x1, z1, x2, z2):
    dx, dz = x2 - x1, z2 - z1
    length2 = dx * dx + dz * dz
    if length2 <= 1e-12:
        return math.hypot(px - x1, pz - z1)
    t = ((px - x1) * dx + (pz - z1) * dz) / length2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x1 + t * dx), pz - (z1 + t * dz))


def polygon_centroid(polygon):
    if not polygon:
        return None
    area = 0.0
    cx = 0.0
    cz = 0.0
    count = len(polygon)
    for index in range(count):
        x1, z1 = polygon[index]
        x2, z2 = polygon[(index + 1) % count]
        cross = x1 * z2 - x2 * z1
        area += cross
        cx += (x1 + x2) * cross
        cz += (z1 + z2) * cross
    if abs(area) < 1e-9:
        xs = [point[0] for point in polygon]
        zs = [point[1] for point in polygon]
        return [sum(xs) / len(xs), sum(zs) / len(zs)]
    area *= 0.5
    return [cx / (6.0 * area), cz / (6.0 * area)]


def polygon_bounds(points):
    if not points:
        return None
    xs = [point[0] for point in points]
    zs = [point[1] for point in points]
    return [min(xs), min(zs), max(xs), max(zs)]


def _merge_bounds(bounds_list):
    present = [item for item in bounds_list if item]
    if not present:
        return None
    return [
        min(item[0] for item in present),
        min(item[1] for item in present),
        max(item[2] for item in present),
        max(item[3] for item in present),
    ]


# --------------------------------------------------------------------------- #
# scenes
# --------------------------------------------------------------------------- #
def _scene_records_from_json(raw):
    """Yield ``(scene_id_hint, record)`` from the common scenes.json shapes."""
    if isinstance(raw, list):
        for index, item in enumerate(raw):
            if isinstance(item, dict):
                yield item.get("scene_id") or item.get("id"), item
        return
    if not isinstance(raw, dict):
        return
    if isinstance(raw.get("scenes"), list):
        for index, item in enumerate(raw["scenes"]):
            if isinstance(item, dict):
                yield item.get("scene_id") or item.get("id"), item
        return
    if isinstance(raw.get("scenes"), dict):
        for key, item in raw["scenes"].items():
            if isinstance(item, dict):
                yield key, item
        return
    if "rooms" in raw or "floors" in raw:
        yield raw.get("scene_id") or raw.get("id"), raw
        return
    # keyed object: {"scene_a": {"rooms": [...]}, ...}
    for key, item in raw.items():
        if isinstance(item, dict) and ("rooms" in item or "floors" in item):
            yield key, item


def _load_json_or_jsonl(path):
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return None
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if path.suffix.lower() != ".jsonl":
                raise
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def discover_sibling(base, names):
    for name in names:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _manifest_for_scene_record(record, base):
    """Resolve a ``scene_manifest`` / ``scene_dir`` pointer in scenes.json."""
    for key in ("scene_manifest", "manifest", "scene3d_manifest"):
        value = record.get(key)
        if isinstance(value, str):
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            if candidate.is_file():
                return candidate
    for key in ("scene_dir", "dir", "directory", "path"):
        value = record.get(key)
        if not isinstance(value, str):
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.is_dir():
            for relative in (
                "build/scene3d_manifest.json",
                "scene3d_manifest.json",
            ):
                nested = candidate / relative
                if nested.is_file():
                    return nested
        elif candidate.is_file() and candidate.suffix.lower() == ".json":
            return candidate
    return None


def load_scene_file(path, warnings):
    base = path if path.is_dir() else path.parent
    if path.is_dir():
        candidate = discover_sibling(path, (
            "scenes.json", "scenes.jsonl", "scene_manifests.json", "scene.json",
        ))
        if candidate is None:
            for child in sorted(path.iterdir()):
                if child.is_dir():
                    candidate = discover_sibling(child, ("scenes.json",))
                    if candidate:
                        break
        if candidate is None:
            raise FileNotFoundError(
                f"no scenes.json / scenes.jsonl under {path}"
            )
        path = candidate
    raw = _load_json_or_jsonl(path)
    records = {}
    for hint, record in _scene_records_from_json(raw):
        if not isinstance(record, dict):
            continue
        scene_id = str(
            record.get("scene_id")
            or record.get("id")
            or hint
            or f"scene_{len(records)}"
        )
        record = dict(record)
        record["scene_id"] = scene_id
        if not record.get("rooms"):
            manifest_path = _manifest_for_scene_record(record, path.parent)
            if manifest_path is not None:
                try:
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as error:
                    warnings.append(
                        f"cannot read scene manifest {manifest_path}: {error}"
                    )
                else:
                    if isinstance(manifest, dict):
                        record = {**manifest, **record, "scene_id": scene_id}
        records[scene_id] = record
    return records, str(path.resolve())


def load_scene_root(scene_root, needed_ids, warnings):
    root = Path(scene_root).expanduser()
    found = {}
    if not root.is_dir():
        warnings.append(f"scene root not found: {root}")
        return found
    for scene_id in needed_ids:
        candidates = (
            root / scene_id / "build" / "scene3d_manifest.json",
            root / scene_id / "scene3d_manifest.json",
            root / scene_id / "build" / "scene_manifest.json",
        )
        for candidate in candidates:
            if candidate.is_file():
                try:
                    record = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    warnings.append(f"cannot read scene manifest {candidate}: {error}")
                    continue
                if isinstance(record, dict):
                    record = dict(record)
                    record.setdefault("scene_id", scene_id)
                    found[scene_id] = record
                break
    return found


def load_scenes_from_run_json(path, warnings):
    """Use ``run.json`` scene records as a geometry fallback.

    The benchmark's ``run.json`` lists each scene's ``build_dir``.  When the
    evaluator did not write a combined ``scenes.json``, the exported
    ``scene3d_manifest.json`` in that directory still provides polygons and
    floors for the replay.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        warnings.append(f"cannot read run summary {path}: {error}")
        return {}
    found = {}
    for record in raw.get("scenes") or []:
        if not isinstance(record, dict):
            continue
        scene_id = record.get("scene_id")
        if not scene_id:
            continue
        scene_id = str(scene_id)
        build_dir = record.get("build_dir")
        candidates = []
        if build_dir:
            base = Path(str(build_dir)).expanduser()
            if not base.is_absolute():
                base = path.parent / base
            candidates.extend((
                base / "scene3d_manifest.json",
                base / "build" / "scene3d_manifest.json",
            ))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                manifest = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                warnings.append(f"cannot read scene manifest {candidate}: {error}")
                continue
            if isinstance(manifest, dict):
                manifest = dict(manifest)
                manifest["scene_id"] = scene_id
                found[scene_id] = manifest
            break
        else:
            warnings.append(
                f"run.json scene {scene_id}: no scene3d_manifest.json under "
                f"{build_dir or '(no build_dir)'}"
            )
    return found


def _ensure_floor(floors, floor_id, elevation=None, name=None):
    floor_id = str(floor_id)
    for floor in floors:
        if floor["id"] == floor_id:
            if floor.get("elevation") is None and elevation is not None:
                floor["elevation"] = elevation
            return floor
    floor = {
        "id": floor_id,
        "elevation": elevation,
        "height": None,
        "name": name or floor_id,
    }
    floors.append(floor)
    return floor


def _source_floor_point(value, mode):
    """Point in the source floor plane (``[x, z]`` for glTF, ``[x, y]`` for Blender)."""
    if isinstance(value, dict):
        for key in ("position", "point", "center", "translation", "xyz"):
            if key in value:
                return _source_floor_point(value[key], mode)
        x = as_float(value.get("x"))
        second = as_float(value.get("z", value.get("y")))
        if x is not None and second is not None:
            return [x, second]
        return None
    if not isinstance(value, (list, tuple)):
        return None
    numbers = [as_float(item) for item in value]
    if len(numbers) >= 3:
        x, y, z = numbers[0], numbers[1], numbers[2]
        if x is None or y is None or z is None:
            return None
        return [x, y] if mode == "blender" else [x, z]
    if len(numbers) == 2:
        x, second = numbers
        if x is None or second is None:
            return None
        return [x, second]
    return None


def normalize_scene(raw, warnings, mode="gltf"):
    scene_id = str(raw.get("scene_id") or raw.get("id") or "scene")
    floors = []
    for item in raw.get("floors") or raw.get("storeys") or []:
        if not isinstance(item, dict):
            continue
        floor_id = item.get("id") or item.get("floor_id") or item.get("name")
        if floor_id is None:
            continue
        floor = _ensure_floor(
            floors,
            floor_id,
            elevation=as_float(item.get("elevation", item.get("y"))),
            name=item.get("name"),
        )
        floor["height"] = as_float(item.get("height"))
        floor["slab_thickness"] = as_float(item.get("slab_thickness"))
        floor["top"] = as_float(item.get("top"))
        floor["name"] = item.get("name") or floor["name"]

    rooms = []
    for index, item in enumerate(raw.get("rooms") or []):
        if not isinstance(item, dict):
            continue
        room_id = str(item.get("id") or item.get("room_id") or f"room_{index}")
        polygon = polygon2(
            item.get("polygon")
            or item.get("footprint")
            or item.get("outline")
            or item.get("floor_polygon")
            or item.get("poly")
            or item.get("vertices")
            or item.get("points")
        )
        if not polygon:
            center = _source_floor_point(
                item.get("center") or item.get("position"), mode
            )
            size = item.get("size")
            if center is not None and isinstance(size, (list, tuple)) and len(size) >= 2:
                width = as_float(size[0])
                depth = as_float(size[1])
                if width and depth:
                    half_w, half_d = abs(width) / 2.0, abs(depth) / 2.0
                    polygon = [
                        [center[0] - half_w, center[1] - half_d],
                        [center[0] + half_w, center[1] - half_d],
                        [center[0] + half_w, center[1] + half_d],
                        [center[0] - half_w, center[1] + half_d],
                    ]
        if not polygon:
            warnings.append(f"scene {scene_id}: room {room_id} has no usable polygon")
            continue
        floor_id = item.get("floor_id", item.get("floor", item.get("level")))
        elevation = as_float(item.get("floor_elevation"))
        if floor_id is None:
            if elevation is not None:
                floor_id = f"f{len(floors)}"
            elif floors:
                floor_id = floors[0]["id"]
            else:
                floor_id = "f0"
        floor_id = str(floor_id)
        _ensure_floor(floors, floor_id, elevation=elevation)
        rooms.append({
            "id": room_id,
            "type": item.get("type") or item.get("room_type"),
            "floor_id": floor_id,
            "polygon": polygon,
            "elevation": elevation,
            "center": polygon_centroid(polygon),
        })

    for floor in floors:
        floor["rooms"] = [room["id"] for room in rooms if room["floor_id"] == floor["id"]]

    voids = []
    for index, item in enumerate(raw.get("voids") or raw.get("holes") or []):
        if not isinstance(item, dict):
            continue
        polygon = polygon2(
            item.get("polygon") or item.get("footprint")
            or item.get("outline") or item.get("poly")
        )
        if not polygon:
            continue
        floor_id = item.get("floor_id", item.get("floor"))
        voids.append({
            "id": str(item.get("id") or f"void_{index}"),
            "floor_id": None if floor_id is None else str(floor_id),
            "polygon": polygon,
            "label": item.get("label") or item.get("type") or "void",
        })

    portals = []
    for index, item in enumerate(
        raw.get("portals") or raw.get("doors") or raw.get("doorways") or []
    ):
        if not isinstance(item, dict):
            continue
        point = _source_floor_point(
            item.get("position") or item.get("center") or item.get("point"),
            mode,
        )
        if point is None:
            polygon = polygon2(item.get("polygon") or item.get("outline"))
            point = polygon_centroid(polygon)
        if point is None:
            continue
        radius = as_float(item.get("radius", item.get("width", 0.9)))
        portals.append({
            "id": str(item.get("id") or f"door_{index}"),
            "floor_id": (
                None if item.get("floor_id") is None else str(item.get("floor_id"))
            ),
            "position": [point[0], point[1]],
            "radius": 0.9 if radius is None else abs(radius),
            "label": item.get("label") or item.get("type") or "door",
        })

    bounds = _merge_bounds([
        polygon_bounds(room["polygon"]) for room in rooms
    ] + [
        polygon_bounds(void["polygon"]) for void in voids
    ])
    raw_bounds = raw.get("bounds")
    if isinstance(raw_bounds, dict):
        low = _source_floor_point(raw_bounds.get("min"), mode)
        high = _source_floor_point(raw_bounds.get("max"), mode)
        if low and high:
            bounds = [low[0], low[1], high[0], high[1]]
    elif isinstance(raw_bounds, (list, tuple)) and len(raw_bounds) == 2:
        low = _source_floor_point(raw_bounds[0], mode)
        high = _source_floor_point(raw_bounds[1], mode)
        if low and high:
            bounds = [low[0], low[1], high[0], high[1]]

    floors.sort(key=lambda floor: (
        floor["elevation"] is None,
        floor["elevation"] if floor["elevation"] is not None else 0.0,
        floor["id"],
    ))
    return {
        "scene_id": scene_id,
        "rooms": rooms,
        "floors": floors,
        "voids": voids,
        "portals": portals,
        "bounds": bounds,
        "coordinate_system": raw.get("coordinate_system"),
        "warnings": [],
    }


def apply_coordinate_mode(scene, mode):
    if mode != "blender":
        return scene
    for room in scene["rooms"]:
        room["polygon"] = blender_polygon_to_gltf(room["polygon"])
        center = polygon_centroid(room["polygon"])
        room["center"] = center
    for void in scene["voids"]:
        void["polygon"] = blender_polygon_to_gltf(void["polygon"])
    for portal in scene["portals"]:
        x, z = portal["position"]
        portal["position"] = [x, -z]
    if scene.get("bounds"):
        min_x, min_z, max_x, max_z = scene["bounds"]
        scene["bounds"] = [min_x, -max_z, max_x, -min_z]
    return scene


# --------------------------------------------------------------------------- #
# episodes
# --------------------------------------------------------------------------- #
def _normalize_points(value, fallback_y, warnings=None, label="path"):
    if value is None:
        return []
    points = []
    current_y = fallback_y
    for index, item in enumerate(value):
        point = position3(item, fallback_y=current_y)
        if point is None:
            if warnings is not None:
                warnings.append(f"{label}[{index}] is not a point")
            continue
        current_y = point[1]
        points.append(
            [rounded(point[0]), rounded(point[1]), rounded(point[2])]
        )
    return points


def _normalize_reference_path(value, fallback_y, warnings):
    distance = None
    raw_points = value
    if isinstance(value, dict):
        raw_points = (
            value.get("points")
            or value.get("path")
            or value.get("positions")
            or value.get("waypoints")
            or []
        )
        distance = as_float(
            value.get("distance", value.get("length", value.get("path_length")))
        )
    points = _normalize_points(raw_points, fallback_y, warnings, "reference_path.points")
    distance_computed = False
    if distance is None and len(points) >= 2:
        distance = sum(
            math.dist(points[index - 1], points[index])
            for index in range(1, len(points))
        )
        distance_computed = True
    return {
        "points": points,
        "distance": rounded(distance, 5),
        "distance_computed": distance_computed,
    }


def _collision_value(value):
    if isinstance(value, dict):
        for key in ("hit", "collision", "blocked", "occurred"):
            if key in value:
                return as_bool(value[key]) is True
        count = as_int(value.get("count", value.get("num_collisions")))
        return bool(count) if count is not None else False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) > 0
    return as_bool(value) is True


def _normalize_trajectory(value, fallback_y, warnings):
    raw_steps = value
    if isinstance(value, dict):
        raw_steps = (
            value.get("steps")
            or value.get("points")
            or value.get("trajectory")
            or value.get("observations")
            or []
        )
    if not isinstance(raw_steps, (list, tuple)):
        return []
    steps = []
    current_y = fallback_y
    for index, item in enumerate(raw_steps or []):
        if isinstance(item, dict):
            point = position3(
                item.get("position")
                or item.get("pos")
                or item.get("point")
                or item.get("location"),
                fallback_y=current_y,
            )
            yaw = as_float(
                item.get(
                    "yaw",
                    item.get("heading", item.get("yaw_deg", item.get("heading_deg"))),
                )
            )
            action = item.get("action")
            collision_value = item.get("collision")
        else:
            point = position3(item, fallback_y=current_y)
            yaw = None
            action = None
            collision_value = None
        if point is None:
            warnings.append(f"trajectory[{index}] has no position")
            continue
        current_y = point[1]
        step = {
            "position": [rounded(point[0]), rounded(point[1]), rounded(point[2])],
            "yaw": rounded(yaw, 5),
            "action": action,
            "collision": _collision_value(collision_value),
            "floor_id": None,
        }
        steps.append(step)
    return steps


def _wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def infer_yaw_metadata(trajectory, unit=None, convention=None):
    """Infer yaw units and convention from the direction of measured motion.

    The project's ``PointNavEnv`` stores degrees with yaw 0 along +X (the
    manifest convention), while Habitat-style logs use radians with yaw 0
    along -Z.  Comparing the measured horizontal displacement with the heading
    makes the replay independent of an undeclared convention.  Explicit
    ``yaw_unit`` / ``yaw_convention`` fields (or CLI flags) win; when the
    trajectory has no usable motion we default to the project convention.
    """
    explicit_unit = unit if unit in ("degrees", "radians") else None
    explicit_convention = (
        convention if convention in ("manifest", "habitat") else None
    )
    samples = []
    for index in range(1, len(trajectory)):
        previous = trajectory[index - 1]["position"]
        current = trajectory[index]["position"]
        yaw = trajectory[index]["yaw"]
        if yaw is None:
            continue
        action = trajectory[index].get("action")
        if isinstance(action, str) and action.strip().lower() in (
            "left", "right", "turn_left", "turn_right",
        ):
            continue
        dx = current[0] - previous[0]
        dz = current[2] - previous[2]
        if math.hypot(dx, dz) < 0.1:
            continue
        observed = math.degrees(math.atan2(-dz, dx))
        samples.append((yaw, observed))

    units = [explicit_unit] if explicit_unit else ["degrees", "radians"]
    conventions = (
        [explicit_convention] if explicit_convention
        else ["manifest", "habitat"]
    )
    best = None
    for candidate_unit in units:
        for candidate_convention in conventions:
            errors = []
            for yaw, observed in samples:
                yaw_degrees = (
                    yaw if candidate_unit == "degrees"
                    else math.degrees(yaw)
                )
                predicted = yaw_degrees + (
                    90.0 if candidate_convention == "habitat" else 0.0
                )
                errors.append(abs(_wrap_degrees(predicted - observed)))
            mean_error = sum(errors) / len(errors) if errors else None
            median_error = (
                sorted(errors)[len(errors) // 2] if errors else None
            )
            rank = (
                median_error if median_error is not None else 999.0,
                0 if candidate_unit == "degrees" else 1,
                0 if candidate_convention == "manifest" else 1,
            )
            if best is None or rank < best[0]:
                best = (
                    rank,
                    candidate_unit,
                    candidate_convention,
                    mean_error,
                    median_error,
                )
    if best is None:
        return (explicit_unit or "degrees", explicit_convention or "manifest", None, 0)
    _, best_unit, best_convention, mean_error, median_error = best
    if (
        median_error is not None
        and median_error > 75.0
        and explicit_unit is None
        and explicit_convention is None
    ):
        # The motion is too noisy (collisions, teleports) to override the
        # project defaults.
        return ("degrees", "manifest", mean_error, len(samples))
    return (best_unit, best_convention, mean_error, len(samples))


def _floor_at_y(y, floors):
    candidates = [
        floor for floor in floors
        if floor.get("elevation") is not None
    ]
    if y is None or not candidates:
        return None
    nearest = min(candidates, key=lambda floor: abs(floor["elevation"] - y))
    if abs(nearest["elevation"] - y) <= 0.6:
        return nearest
    below = [
        floor for floor in candidates
        if floor["elevation"] <= y + 0.25
    ]
    if below:
        return max(below, key=lambda floor: floor["elevation"])
    return nearest if abs(nearest["elevation"] - y) <= 2.0 else None


def _floor_id_at_y(y, floors):
    floor = _floor_at_y(y, floors)
    return floor["id"] if floor else None


def _room_at(x, z, rooms, tolerance=0.0):
    matches = [
        room for room in rooms
        if point_in_polygon(x, z, room["polygon"], tolerance=tolerance)
    ]
    if not matches:
        return None
    # Prefer the smallest containing room (avoids a big open-plan polygon
    # swallowing a nested room).
    matches.sort(key=lambda room: abs(_polygon_area(room["polygon"])))
    return matches[0]


def _polygon_area(polygon):
    area = 0.0
    count = len(polygon)
    for index in range(count):
        x1, z1 = polygon[index]
        x2, z2 = polygon[(index + 1) % count]
        area += x1 * z2 - x2 * z1
    return area / 2.0


def _near_polygon(x, z, polygon, tolerance):
    if point_in_polygon(x, z, polygon, tolerance=0.0):
        return True
    if len(polygon) < 2:
        return False
    count = len(polygon)
    for index in range(count):
        x1, z1 = polygon[index]
        x2, z2 = polygon[(index + 1) % count]
        if _point_segment_distance(x, z, x1, z1, x2, z2) <= tolerance:
            return True
    return False


def _supported_by_rooms(x, z, rooms, tolerance):
    return any(_near_polygon(x, z, room["polygon"], tolerance) for room in rooms)


def _rooms_on_floor(scene, floor_id):
    if floor_id is None:
        return list(scene["rooms"])
    return [
        room for room in scene["rooms"] if room["floor_id"] == str(floor_id)
    ]


def _portal_near(scene, x, z, floor_id, extra=0.0):
    best = None
    for portal in scene.get("portals", []):
        px, pz = portal["position"]
        if portal.get("floor_id") is not None and floor_id is not None:
            if str(portal["floor_id"]) != str(floor_id):
                continue
        distance = math.hypot(px - x, pz - z)
        if distance <= portal["radius"] + extra:
            if best is None or distance < best[0]:
                best = (distance, portal)
    return best[1] if best else None


def _segment_samples(a, b, count=9):
    ax, ay, az = a
    bx, by, bz = b
    for index in range(1, count):
        t = index / float(count)
        yield (
            ax + (bx - ax) * t,
            ay + (by - ay) * t,
            az + (bz - az) * t,
            t,
        )


def classify_episode(episode, scene):
    """Attach floor ids, events and a primary failure category in place."""
    floors = scene["floors"]
    rooms = scene["rooms"]
    warnings = episode["warnings"]
    start = episode["start"]
    goal = episode["goal"]
    if start is not None:
        inferred = _floor_id_at_y(start[1], floors)
        if episode["start_floor"] is None:
            episode["start_floor"] = inferred
        elif inferred is not None and str(episode["start_floor"]) != str(inferred):
            warnings.append(
                f"start_floor {episode['start_floor']} disagrees with the "
                f"start elevation (inferred {inferred})"
            )
    if goal is not None:
        inferred = _floor_id_at_y(goal[1], floors)
        if episode["goal_floor"] is None:
            episode["goal_floor"] = inferred
        elif inferred is not None and str(episode["goal_floor"]) != str(inferred):
            warnings.append(
                f"goal_floor {episode['goal_floor']} disagrees with the "
                f"goal elevation (inferred {inferred})"
            )

    active_floor = episode["start_floor"]
    for step in episode["trajectory"]:
        floor_id = _floor_id_at_y(step["position"][1], floors)
        if floor_id is None:
            floor_id = active_floor
        else:
            active_floor = floor_id
        step["floor_id"] = floor_id

    path_floor_ids = []
    active_floor = episode["start_floor"]
    for point in episode["reference_path"]["points"]:
        floor_id = _floor_id_at_y(point[1], floors)
        if floor_id is None:
            floor_id = active_floor
        else:
            active_floor = floor_id
        path_floor_ids.append(floor_id)
    episode["reference_path"]["floor_ids"] = path_floor_ids

    events = []
    if start is not None:
        events.append({
            "type": "start",
            "step": 0,
            "position": start,
            "floor_id": episode["start_floor"],
            "label": "start",
        })
    if goal is not None:
        events.append({
            "type": "goal",
            "step": max(0, len(episode["trajectory"]) - 1),
            "position": goal,
            "floor_id": episode["goal_floor"],
            "label": "goal",
        })

    trajectory = episode["trajectory"]
    collision_steps = [
        index for index, step in enumerate(trajectory) if step["collision"]
    ]

    # Collision points are useful even for successful episodes.
    seen_collision = set()
    for index in collision_steps:
        point = trajectory[index]["position"]
        key = tuple(round(value, 2) for value in point)
        if key in seen_collision:
            continue
        seen_collision.add(key)
        events.append({
            "type": "collision",
            "step": index,
            "position": point,
            "floor_id": trajectory[index]["floor_id"],
            "label": f"collision ({episode['collisions']} total)"
            if episode["collisions"] is not None
            else "collision",
        })

    coverage = None
    if trajectory and rooms:
        supported = 0
        for step in trajectory:
            point = step["position"]
            floor_rooms = _rooms_on_floor(scene, step["floor_id"])
            if floor_rooms and _supported_by_rooms(
                point[0], point[2], floor_rooms, DEFAULT_SUPPORT_TOLERANCE
            ):
                supported += 1
        coverage = supported / float(len(trajectory))
    geometry_reliable = coverage is not None and coverage >= 0.5

    # Through-wall / leaving the floor polygons, and illegal room crossings.
    if geometry_reliable and len(trajectory) >= 2:
        through_points = []
        crossing_points = []
        for index in range(1, len(trajectory)):
            previous = trajectory[index - 1]["position"]
            current = trajectory[index]["position"]
            floor_id = trajectory[index]["floor_id"] or trajectory[index - 1]["floor_id"]
            floor_rooms = _rooms_on_floor(scene, floor_id)
            if not floor_rooms:
                continue
            midpoint = [
                (previous[0] + current[0]) / 2.0,
                (previous[1] + current[1]) / 2.0,
                (previous[2] + current[2]) / 2.0,
            ]
            mid_supported = _supported_by_rooms(
                midpoint[0], midpoint[2], floor_rooms, DEFAULT_SUPPORT_TOLERANCE
            )
            if not mid_supported:
                through_points.append((index, midpoint))
                continue
            previous_room = _room_at(previous[0], previous[2], floor_rooms)
            current_room = _room_at(current[0], current[2], floor_rooms)
            if (
                scene.get("portals")
                and previous_room is not None
                and current_room is not None
                and previous_room["id"] != current_room["id"]
            ):
                # Without door metadata an adjacent-room transition is
                # indistinguishable from a legitimate doorway, so only
                # boundary crossings are judged when portals are declared.
                crossing = midpoint
                for sample in _segment_samples(previous, current):
                    sample_room = _room_at(sample[0], sample[2], floor_rooms)
                    if sample_room is not None and sample_room["id"] != previous_room["id"]:
                        crossing = [sample[0], sample[1], sample[2]]
                        break
                portal = _portal_near(
                    scene, crossing[0], crossing[2], floor_id
                )
                if portal is None:
                    crossing_points.append((index, crossing))
        for index, point in through_points[:6]:
            events.append({
                "type": "through_wall",
                "step": index,
                "position": [rounded(point[0]), rounded(point[1]), rounded(point[2])],
                "floor_id": trajectory[index]["floor_id"],
                "label": "trajectory leaves the room polygons",
            })
        for index, point in crossing_points[:6]:
            events.append({
                "type": "through_wall",
                "step": index,
                "position": [rounded(point[0]), rounded(point[1]), rounded(point[2])],
                "floor_id": trajectory[index]["floor_id"],
                "label": "room boundary crossed away from a door",
            })

    # Cross-void: any sampled trajectory point inside a declared slab hole.
    if geometry_reliable and len(trajectory) >= 2:
        void_hits = []
        for index in range(1, len(trajectory)):
            previous = trajectory[index - 1]["position"]
            current = trajectory[index]["position"]
            for sample in _segment_samples(previous, current, count=9):
                x, y, z, _ = sample
                floor_id = _floor_id_at_y(y, floors) or trajectory[index]["floor_id"]
                for void in scene["voids"]:
                    if (
                        void.get("floor_id") is not None
                        and floor_id is not None
                        and str(void["floor_id"]) != str(floor_id)
                    ):
                        continue
                    if point_in_polygon(x, z, void["polygon"], tolerance=0.02):
                        void_hits.append(
                            (
                                index,
                                [rounded(x), rounded(y), rounded(z)],
                                floor_id,
                                void["id"],
                            )
                        )
                        break
                else:
                    continue
                break
        # One marker per continuous crossing: consecutive trajectory segments
        # through the same hole are the same event.
        collapsed = []
        last_hit = {}
        for index, point, floor_id, void_id in void_hits:
            previous_index = last_hit.get(void_id)
            if previous_index is not None and index - previous_index <= 1:
                last_hit[void_id] = index
                continue
            collapsed.append((index, point, floor_id, void_id))
            last_hit[void_id] = index
        for index, point, floor_id, void_id in collapsed[:6]:
            events.append({
                "type": "cross_void",
                "step": index,
                "position": point,
                "floor_id": floor_id,
                "label": f"trajectory crosses void {void_id}",
                "void": void_id,
            })

    # Floor teleports: a vertical change no stair walk can produce in one step.
    if len(trajectory) >= 2:
        for index in range(1, len(trajectory)):
            previous = trajectory[index - 1]
            current = trajectory[index]
            dy = current["position"][1] - previous["position"][1]
            floor_changed = (
                previous["floor_id"] is not None
                and current["floor_id"] is not None
                and previous["floor_id"] != current["floor_id"]
            )
            horizontal = math.hypot(
                current["position"][0] - previous["position"][0],
                current["position"][2] - previous["position"][2],
            )
            if abs(dy) >= 1.0 and (floor_changed or abs(dy) >= 1.8):
                direction = "down" if dy < 0 else "up"
                events.append({
                    "type": "floor_jump",
                    "step": index,
                    "position": [
                        rounded((previous["position"][0] + current["position"][0]) / 2.0),
                        rounded((previous["position"][1] + current["position"][1]) / 2.0),
                        rounded((previous["position"][2] + current["position"][2]) / 2.0),
                    ],
                    "floor_id": current["floor_id"],
                    "label": (
                        f"floor changed {previous['floor_id'] or '?'} -> "
                        f"{current['floor_id'] or '?'} in one step "
                        f"({direction} {abs(dy):.2f} m, horizontal {horizontal:.2f} m)"
                    ),
                })

    # Stuck clusters: several collisions with almost no displacement.
    stuck_added = False
    if trajectory:
        for start_index in collision_steps:
            window_collisions = [
                index for index in collision_steps
                if start_index <= index <= start_index + 10
            ]
            if len(window_collisions) < 3:
                continue
            first = trajectory[start_index]["position"]
            last_index = min(start_index + 10, len(trajectory) - 1)
            last = trajectory[last_index]["position"]
            displacement = math.hypot(last[0] - first[0], last[2] - first[2])
            if displacement > 0.6:
                continue
            positions = [trajectory[index]["position"] for index in window_collisions]
            centre = [
                sum(point[0] for point in positions) / len(positions),
                sum(point[1] for point in positions) / len(positions),
                sum(point[2] for point in positions) / len(positions),
            ]
            floor_id = trajectory[window_collisions[0]]["floor_id"]
            portal = _portal_near(scene, centre[0], centre[2], floor_id, extra=0.6)
            event_type = "stuck_door" if portal is not None else "stuck"
            label = (
                f"{len(window_collisions)} collisions within "
                f"{displacement:.2f} m near {portal['id']}"
                if portal is not None
                else f"{len(window_collisions)} collisions within {displacement:.2f} m"
            )
            events.append({
                "type": event_type,
                "step": window_collisions[0],
                "position": [rounded(value) for value in centre],
                "floor_id": floor_id,
                "label": label,
                "portal": portal["id"] if portal is not None else None,
            })
            stuck_added = True
            break
        if (
            not stuck_added
            and len(collision_steps) >= 4
            and len(trajectory) >= 2
        ):
            first = trajectory[0]["position"]
            last = trajectory[-1]["position"]
            span = math.hypot(last[0] - first[0], last[2] - first[2])
            if span < 0.8:
                positions = [trajectory[index]["position"] for index in collision_steps]
                centre = [
                    sum(point[0] for point in positions) / len(positions),
                    sum(point[1] for point in positions) / len(positions),
                    sum(point[2] for point in positions) / len(positions),
                ]
                events.append({
                    "type": "stuck",
                    "step": collision_steps[0],
                    "position": [rounded(value) for value in centre],
                    "floor_id": trajectory[collision_steps[0]]["floor_id"],
                    "label": (
                        f"{len(collision_steps)} collisions with "
                        f"{span:.2f} m total displacement"
                    ),
                })

    events.sort(key=lambda event: (event["step"], event["type"]))
    episode["events"] = events
    episode["geometry_coverage"] = rounded(coverage)
    episode["geometry_reliable"] = geometry_reliable

    event_types = {event["type"] for event in events}
    reason_category = category_from_reason(episode["failure_reason"])
    episode["reason_category"] = reason_category

    if episode["success"] is True:
        category = "success"
    elif reason_category in SPECIFIC_CATEGORIES:
        category = reason_category
    elif "cross_void" in event_types:
        category = "cross_void"
    elif "through_wall" in event_types:
        category = "through_wall"
    elif "floor_jump" in event_types:
        category = "floor_jump"
    elif "stuck_door" in event_types or reason_category == "stuck_door":
        category = "stuck_door"
    elif "stuck" in event_types or reason_category == "stuck":
        category = "stuck"
    elif reason_category == "collision":
        category = "collision"
    elif episode["collisions"]:
        category = "collision"
    elif reason_category == "timeout" or episode["timeout"]:
        category = "timeout"
    else:
        category = "other"
    episode["failure_category"] = category

    if episode["success"] is False and not episode["failure_reason"]:
        warnings.append("failure_reason is empty")
    if episode["success"] is True and episode["failure_reason"]:
        warnings.append("success is true but failure_reason is set")
    return episode


def category_from_reason(reason):
    text = reason_text(reason).lower()
    if not text:
        return None
    for category, keywords in REASON_KEYWORDS:
        for keyword in keywords:
            if keyword in text:
                return category
    return "other"


def reason_text(reason):
    if reason is None:
        return ""
    if isinstance(reason, dict):
        for key in ("code", "type", "category", "reason", "message", "detail"):
            value = reason.get(key)
            if value:
                return str(value)
        return json.dumps(reason, ensure_ascii=False, sort_keys=True)
    return str(reason)


def normalize_episode(raw, index, warnings):
    if not isinstance(raw, dict):
        raise ValueError(f"line {index} is not a JSON object")
    scene_id = str(raw.get("scene_id") or raw.get("scene") or "unknown")
    episode_id = raw.get("episode_id", raw.get("id", index))
    episode_warnings = []

    start = position3(
        raw.get("start", raw.get("start_position")), fallback_y=0.0
    )
    goal = position3(
        raw.get("goal", raw.get("goal_position")), fallback_y=0.0
    )
    fallback_y = start[1] if start is not None else 0.0
    reference_path = _normalize_reference_path(
        raw.get("reference_path")
        or raw.get("reference")
        or raw.get("shortest_path")
        or raw.get("path"),
        fallback_y,
        episode_warnings,
    )
    if not reference_path["points"]:
        episode_warnings.append("no reference_path points")
    if start is None and reference_path["points"]:
        start = reference_path["points"][0][:]
        episode_warnings.append("start missing; using the first reference point")
    if goal is None and reference_path["points"]:
        goal = reference_path["points"][-1][:]
        episode_warnings.append("goal missing; using the last reference point")
    if start is None:
        episode_warnings.append("start missing")
    if goal is None:
        episode_warnings.append("goal missing")

    trajectory = _normalize_trajectory(
        raw.get("trajectory") or raw.get("observations"),
        fallback_y,
        episode_warnings,
    )
    raw_actions = raw.get("actions")
    if isinstance(raw_actions, (list, tuple)):
        for index, step in enumerate(trajectory):
            if step["action"] is None and index < len(raw_actions):
                step["action"] = raw_actions[index]
    if not trajectory:
        episode_warnings.append("no trajectory points; showing the reference path only")
    elif len(trajectory) < 2:
        episode_warnings.append("trajectory has fewer than two points")

    yaw_unit, yaw_convention, yaw_error, yaw_samples = infer_yaw_metadata(
        trajectory,
        unit=yaw_unit_name(raw.get("yaw_unit")),
        convention=yaw_convention_name(
            raw.get("yaw_convention", raw.get("yaw_frame"))
        ),
    )

    start_floor = raw.get("start_floor")
    goal_floor = raw.get("goal_floor")
    start_floor = None if start_floor is None else str(start_floor)
    goal_floor = None if goal_floor is None else str(goal_floor)

    success = as_bool(raw.get("success"))
    if success is None:
        episode_warnings.append("success is missing")
    spl = as_float(raw.get("spl", raw.get("success_weighted_path_length")))
    raw_collisions = raw.get("collisions", raw.get("collision_count"))
    if isinstance(raw_collisions, (list, tuple)):
        collisions = len(raw_collisions)
    else:
        collisions = as_int(raw_collisions)
    timeout = as_bool(raw.get("timeout"))
    if timeout is None:
        timeout = False
    if collisions is None and trajectory:
        collisions = sum(1 for step in trajectory if step["collision"])
        episode_warnings.append(
            f"collisions missing; using {collisions} collision steps"
        )

    trajectory_distance = None
    if len(trajectory) >= 2:
        trajectory_distance = sum(
            math.dist(
                trajectory[position - 1]["position"],
                trajectory[position]["position"],
            )
            for position in range(1, len(trajectory))
        )
    final_distance = None
    if goal is not None and trajectory:
        final_distance = math.dist(trajectory[-1]["position"], goal)

    spl_computed = False
    if spl is None and success is not None:
        reference_distance = reference_path["distance"]
        if reference_distance is not None and trajectory_distance is not None:
            denominator = max(reference_distance, trajectory_distance)
            spl = (
                (1.0 if success else 0.0) * reference_distance / denominator
                if denominator > 0
                else (1.0 if success else 0.0)
            )
            spl_computed = True
        elif reference_distance is None and success:
            spl = 1.0 if not trajectory else 0.0
            spl_computed = True
    if success is False and spl is None:
        spl = 0.0
        spl_computed = True

    episode = {
        "key": f"{scene_id}::{episode_id}",
        "scene_id": scene_id,
        "episode_id": str(episode_id),
        "seed": raw.get("seed"),
        "start": start,
        "goal": goal,
        "start_floor": start_floor,
        "goal_floor": goal_floor,
        "reference_path": reference_path,
        "trajectory": trajectory,
        "yaw_unit": yaw_unit,
        "yaw_convention": yaw_convention,
        "yaw_inference": {
            "movement_samples": yaw_samples,
            "mean_direction_error_deg": rounded(yaw_error, 2),
        },
        "success": success,
        "spl": rounded(spl, 5),
        "spl_computed": spl_computed,
        "collisions": collisions,
        "timeout": timeout,
        "failure_reason": raw.get("failure_reason"),
        "failure_category": None,
        "events": [],
        "metrics": {
            "steps": len(trajectory),
            "collision_steps": sum(1 for step in trajectory if step["collision"]),
            "trajectory_distance": rounded(trajectory_distance, 4),
            "reference_distance": reference_path["distance"],
            "final_distance_to_goal": rounded(final_distance, 4),
            "path_efficiency": (
                rounded(
                    reference_path["distance"] / max(reference_path["distance"], trajectory_distance),
                    4,
                )
                if (
                    reference_path["distance"] is not None
                    and trajectory_distance is not None
                    and max(reference_path["distance"], trajectory_distance) > 0
                )
                else None
            ),
        },
        "has_trajectory": bool(trajectory),
        "warnings": episode_warnings,
    }
    return episode


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _episode_spl(episode):
    return episode["spl"]


def summarize_episodes(episodes):
    def summary(group):
        total = len(group)
        successes = sum(1 for episode in group if episode["success"] is True)
        failures = sum(1 for episode in group if episode["success"] is False)
        unknown = total - successes - failures
        spl_values = [
            value for value in (_episode_spl(episode) for episode in group)
            if value is not None
        ]
        reference_distances = [
            episode["metrics"]["reference_distance"] for episode in group
            if episode["metrics"]["reference_distance"] is not None
        ]
        trajectory_distances = [
            episode["metrics"]["trajectory_distance"] for episode in group
            if episode["metrics"]["trajectory_distance"] is not None
        ]
        categories = {}
        for episode in group:
            category = episode["failure_category"] or "other"
            categories[category] = categories.get(category, 0) + 1
        return {
            "episodes": total,
            "successes": successes,
            "failures": failures,
            "unknown_results": unknown,
            "success_rate": rounded(successes / total, 4) if total else None,
            "spl_mean": rounded(sum(spl_values) / len(spl_values), 4)
            if spl_values else None,
            "spl_episodes": len(spl_values),
            "mean_reference_distance": rounded(
                sum(reference_distances) / len(reference_distances), 4
            ) if reference_distances else None,
            "mean_trajectory_distance": rounded(
                sum(trajectory_distances) / len(trajectory_distances), 4
            ) if trajectory_distances else None,
            "collisions": sum(
                episode["collisions"]
                if episode["collisions"] is not None
                else episode["metrics"]["collision_steps"]
                for episode in group
            ),
            "collision_steps": sum(
                episode["metrics"]["collision_steps"] for episode in group
            ),
            "timeouts": sum(1 for episode in group if episode["timeout"]),
            "failure_categories": dict(sorted(categories.items())),
        }

    by_scene = {}
    for scene_id in sorted({episode["scene_id"] for episode in episodes}):
        by_scene[scene_id] = summary(
            [episode for episode in episodes if episode["scene_id"] == scene_id]
        )
    return {"overall": summary(episodes), "by_scene": by_scene}


def _flatten_scalar_metrics(raw, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                if isinstance(value, dict) and len(out) < 200:
                    _flatten_scalar_metrics(value, key_path, out)
            elif as_float(value) is not None or isinstance(value, (str, bool)):
                out[key_path] = value
    return out


def load_metrics_file(path, warnings):
    if path.is_dir():
        candidate = discover_sibling(
            path, ("metrics.json", "metrics_summary.json", "eval_metrics.json")
        )
        if candidate is None:
            warnings.append(f"no metrics.json under {path}; computing from episodes")
            return None
        path = candidate
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        warnings.append(f"cannot read metrics {path}: {error}")
        return None
    if not isinstance(raw, dict):
        warnings.append(f"metrics {path} is not a JSON object; ignoring")
        return None
    overall = raw.get("overall") or raw.get("summary") or raw.get("metrics") or raw
    if not isinstance(overall, dict):
        overall = raw
    by_scene = (
        raw.get("by_scene")
        or raw.get("scene_summaries")
        or raw.get("scenes")
    )
    if not isinstance(by_scene, dict):
        by_scene = None
    return {
        "source": str(path.resolve()),
        "overall": overall,
        "by_scene": by_scene,
        "flat": _flatten_scalar_metrics(overall),
    }


# --------------------------------------------------------------------------- #
# input discovery
# --------------------------------------------------------------------------- #
def discover_episodes(path, warnings):
    path = Path(path).expanduser()
    if path.is_dir():
        candidates = (
            path / EPISODE_FILENAME,
            path / "episodes" / EPISODE_FILENAME,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        matches = sorted(path.glob(f"*/{EPISODE_FILENAME}"))
        if matches:
            warnings.append(
                f"episodes file not at {path}/{EPISODE_FILENAME}; using {matches[0]}"
            )
            return matches[0]
        if (path / "run.json").is_file():
            raise FileNotFoundError(
                f"{path}/run.json is an evaluation summary without per-step "
                f"trajectories; export {EPISODE_FILENAME} with position, yaw, "
                "action and collision per step"
            )
        raise FileNotFoundError(f"no {EPISODE_FILENAME} under {path}")
    if not path.is_file():
        raise FileNotFoundError(f"episodes file not found: {path}")
    if path.suffix.lower() == ".json":
        try:
            first = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            first = None
        if isinstance(first, dict) and first.get("kind") == "roomgen_pointnav":
            raise FileNotFoundError(
                f"{path} is an evaluation summary without per-step "
                f"trajectories; pass {EPISODE_FILENAME} instead"
            )
    return path


def read_episodes(path, warnings):
    episodes = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as error:
                warnings.append(f"{path.name}:{line_number}: invalid JSON ({error}); skipped")
                continue
            try:
                episodes.append(normalize_episode(raw, line_number, warnings))
            except (TypeError, ValueError) as error:
                warnings.append(f"{path.name}:{line_number}: {error}; skipped")
    if not episodes:
        raise ValueError(f"no usable episodes in {path}")
    return episodes


def _episode_scene_records(episode_raw):
    """Scenes embedded in an episode line are a last-resort fallback."""
    if not isinstance(episode_raw, dict):
        return None
    embedded = episode_raw.get("scene")
    if isinstance(embedded, dict):
        return embedded
    if isinstance(episode_raw.get("rooms"), list):
        return {
            "scene_id": episode_raw.get("scene_id"),
            "rooms": episode_raw.get("rooms"),
            "floors": episode_raw.get("floors") or [],
            "voids": episode_raw.get("voids") or [],
            "portals": episode_raw.get("portals") or [],
        }
    return None


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_site(
    episodes_path,
    out_dir,
    scenes_path=None,
    metrics_path=None,
    scene_root=None,
    title=None,
    coordinate_system=None,
    yaw_unit=None,
    yaw_convention=None,
    quiet=False,
):
    warnings = []
    episodes_path = discover_episodes(episodes_path, warnings)
    base = episodes_path.parent

    if scenes_path is None:
        scenes_path = discover_sibling(base, (
            "scenes.json", "scenes.jsonl", "scene_manifests.json", "scene.json",
        ))
    run_json = discover_sibling(base, ("run.json",))
    if metrics_path is None:
        metrics_path = discover_sibling(base, (
            "metrics.json", "metrics_summary.json", "eval_metrics.json",
        ))
    if metrics_path is None and run_json is not None:
        metrics_path = run_json

    raw_scenes = {}
    scenes_source = None
    if scenes_path is not None:
        raw_scenes, scenes_source = load_scene_file(Path(scenes_path), warnings)

    global_mode = coord_mode(coordinate_system)
    if coordinate_system is None:
        for record in raw_scenes.values():
            if record.get("coordinate_system"):
                global_mode = coord_mode(record["coordinate_system"])
                break

    # Read raw episode lines once so embedded scene records remain available.
    raw_by_line = {}
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                raw_by_line[line_number] = json.loads(text)
            except json.JSONDecodeError:
                pass

    episodes = read_episodes(episodes_path, warnings)
    if yaw_unit is not None:
        override_unit = yaw_unit_name(yaw_unit)
        if override_unit is None:
            warnings.append(f"unknown yaw unit {yaw_unit!r}; keeping inference")
        else:
            for episode in episodes:
                episode["yaw_unit"] = override_unit
    if yaw_convention is not None:
        override_convention = yaw_convention_name(yaw_convention)
        if override_convention is None:
            warnings.append(
                f"unknown yaw convention {yaw_convention!r}; keeping inference"
            )
        else:
            for episode in episodes:
                episode["yaw_convention"] = override_convention

    embedded_scenes = {}
    for line_number, raw in raw_by_line.items():
        embedded = _episode_scene_records(raw)
        if embedded is None:
            continue
        scene_id = str(
            embedded.get("scene_id")
            or raw.get("scene_id")
            or "unknown"
        )
        embedded_scenes.setdefault(scene_id, embedded)

    needed_scene_ids = sorted({episode["scene_id"] for episode in episodes})
    if scene_root is not None:
        for scene_id, record in load_scene_root(
            scene_root, needed_scene_ids, warnings
        ).items():
            raw_scenes.setdefault(scene_id, record)
    if run_json is not None:
        for scene_id, record in load_scenes_from_run_json(
            run_json, warnings
        ).items():
            raw_scenes.setdefault(scene_id, record)
        if scenes_source is None and any(
            scene_id in raw_scenes for scene_id in needed_scene_ids
        ):
            scenes_source = str(Path(run_json).resolve())

    scenes = {}
    for scene_id in needed_scene_ids:
        raw = raw_scenes.get(scene_id) or embedded_scenes.get(scene_id)
        if raw is None:
            scene = {
                "scene_id": scene_id,
                "rooms": [],
                "floors": [],
                "voids": [],
                "portals": [],
                "bounds": None,
                "warnings": [],
            }
            warnings.append(
                f"no scene geometry for {scene_id}; paths are drawn on a blank grid"
            )
        else:
            scene_warnings = []
            scene = normalize_scene(
                {**raw, "scene_id": scene_id}, scene_warnings, mode=global_mode
            )
            scene["warnings"] = scene_warnings
            warnings.extend(scene_warnings)
        if global_mode == "blender":
            apply_coordinate_mode(scene, global_mode)
        scenes[scene_id] = scene

    # Re-normalise episode points into the same frame when needed.
    if global_mode == "blender":
        def convert_point(point):
            return None if point is None else blender_position_to_gltf(point)

        for episode in episodes:
            episode["start"] = convert_point(episode["start"])
            episode["goal"] = convert_point(episode["goal"])
            episode["reference_path"]["points"] = [
                convert_point(point) for point in episode["reference_path"]["points"]
            ]
            for step in episode["trajectory"]:
                step["position"] = convert_point(step["position"])

    # Floors and room metadata are attached before classification.
    for episode in episodes:
        scene = scenes[episode["scene_id"]]
        if not scene["floors"]:
            for floor_id in (
                episode.get("start_floor"),
                episode.get("goal_floor"),
            ):
                if floor_id is not None:
                    _ensure_floor(scene["floors"], floor_id, elevation=None)
            scene["floors"].sort(key=lambda floor: (
                floor["elevation"] is None,
                floor["elevation"] if floor["elevation"] is not None else 0.0,
                floor["id"],
            ))
        classify_episode(episode, scene)
        episode["warnings"] = [
            warning for warning in episode["warnings"] if warning
        ]
        warnings.extend(
            f"{episode['key']}: {warning}" for warning in episode["warnings"]
        )

    # Derived floor lists also need an entry for floors referenced only by an
    # episode (e.g. a missing goal floor in a partial scene).
    for scene in scenes.values():
        for floor in scene["floors"]:
            if floor.get("name") is None:
                floor["name"] = floor["id"]

    computed_metrics = summarize_episodes(episodes)
    provided_metrics = (
        load_metrics_file(Path(metrics_path), warnings)
        if metrics_path is not None else None
    )

    failures = sum(
        1 for episode in episodes
        if episode["failure_category"] not in ("success",)
    )
    data = {
        "kind": "roomgen_nav_replay",
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": title or "RoomGen.Agent PointNav replay",
        "coordinate_system": (
            "gltf_xz_y_up" if global_mode == "gltf"
            else "gltf_xz_y_up (converted from blender z-up)"
        ),
        "source": {
            "episodes": str(episodes_path.resolve()),
            "scenes": scenes_source,
            "metrics": provided_metrics["source"] if provided_metrics else None,
        },
        "categories": CATEGORY_LABELS,
        "metrics": {
            "computed": computed_metrics,
            "provided": provided_metrics,
        },
        "scenes": scenes,
        "episodes": episodes,
        "summary": {
            "episodes": len(episodes),
            "scenes": len(scenes),
            "successes": computed_metrics["overall"]["successes"],
            "failures": computed_metrics["overall"]["failures"],
            "failure_categories": computed_metrics["overall"]["failure_categories"],
            "warnings": len(warnings),
        },
        "warnings": warnings,
    }

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ASSET_FILES:
        source = ASSET_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"replay asset missing: {source}")
        shutil.copy2(source, out_dir / name)

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    data_js = (
        "/* Generated by tools/build_nav_replay.py - do not edit. */\n"
        f"window.NAV_REPLAY_DATA = {payload};\n"
    )
    _atomic_write(out_dir / "replay-data.js", data_js)
    manifest = {
        "kind": "roomgen_nav_replay_site",
        "version": 1,
        "generated_utc": data["generated_utc"],
        "episodes": len(episodes),
        "scenes": len(scenes),
        "successes": computed_metrics["overall"]["successes"],
        "failures": computed_metrics["overall"]["failures"],
        "failure_categories": computed_metrics["overall"]["failure_categories"],
        "warnings": warnings,
        "files": list(ASSET_FILES) + ["replay-data.js", "replay_manifest.json"],
        "source": data["source"],
    }
    _atomic_write(
        out_dir / "replay_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )

    if not quiet:
        print(
            f"nav replay: {len(episodes)} episodes, {len(scenes)} scenes, "
            f"{failures} failures -> {out_dir}",
            file=sys.stderr,
        )
        for warning in warnings[:12]:
            print(f"  warning: {warning}", file=sys.stderr)
        if len(warnings) > 12:
            print(f"  ... {len(warnings) - 12} more warnings", file=sys.stderr)
    return data, out_dir


def _atomic_write(path, text):
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    )
    try:
        with handle:
            handle.write(text)
        written = Path(handle.name)
        os.chmod(written, 0o644)
        written.replace(path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def serve_site(out_dir, host="127.0.0.1", port=8765):
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=str(out_dir), **kwargs
    )
    server = ThreadingHTTPServer((host, port), handler)
    actual_host, actual_port = server.server_address[:2]
    print(
        f"serving {out_dir} at http://{actual_host}:{actual_port}/ "
        "(Ctrl+C to stop)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a static PointNav episode replay site."
    )
    parser.add_argument(
        "run", nargs="?",
        help="evaluation directory or episodes.jsonl (same as --episodes)",
    )
    parser.add_argument(
        "--episodes", default=None,
        help="episodes.jsonl, or an evaluation directory containing it",
    )
    parser.add_argument("--scenes", default=None, help="scenes.json (auto-discovered)")
    parser.add_argument("--metrics", default=None, help="metrics.json (auto-discovered)")
    parser.add_argument(
        "--scene-root", default=None,
        help="directory with <scene_id>/build/scene3d_manifest.json fallbacks",
    )
    parser.add_argument(
        "--out", "--output", dest="out", default="outputs/nav_replay/site",
        help="output directory (default: outputs/nav_replay/site)",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--coordinate-system", choices=("gltf", "habitat", "blender", "z_up"),
        default=None,
        help="input frame; blender z-up is converted to the glTF x/z frame",
    )
    parser.add_argument(
        "--yaw-unit", choices=("auto", "degrees", "radians"), default="auto",
        help="trajectory yaw unit; auto infers it from the motion (default)",
    )
    parser.add_argument(
        "--yaw-convention", choices=("auto", "manifest", "habitat"),
        default="auto",
        help="yaw 0 direction: manifest=+X (project), habitat=-Z",
    )
    parser.add_argument("--serve", action="store_true", help="serve the site after building")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    episodes_path = args.episodes or args.run
    if not episodes_path:
        parser.error("provide an evaluation directory / episodes.jsonl")

    try:
        data, out_dir = build_site(
            episodes_path=episodes_path,
            out_dir=args.out,
            scenes_path=args.scenes,
            metrics_path=args.metrics,
            scene_root=args.scene_root,
            title=args.title,
            coordinate_system=args.coordinate_system,
            yaw_unit=None if args.yaw_unit == "auto" else args.yaw_unit,
            yaw_convention=(
                None if args.yaw_convention == "auto" else args.yaw_convention
            ),
            quiet=args.quiet,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.serve:
        serve_site(out_dir, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
