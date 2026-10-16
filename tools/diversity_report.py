"""Structural diversity of generated houses -- and the 100-seed benchmark.

`layout_gen.py` can produce a thousand specs; the question this tool answers is
whether they are a thousand *different* houses or the same corridor with other
furniture.  It reads specs (a batch root, explicit paths, or seeds it generates)
and measures, per scene:

| metric | of which |
|---|---|
| `rooms`, `room_types`, `floors` | room count, type histogram, storey count |
| `structure` | the **attributed room graph**: one node per room *instance* (function type + storey), one edge per connection (`door` / `open_passage` / `stair` / `ramp`), its cycle rank, and a canonical fingerprint of that graph |
| `shape` | bounding box, **union footprint** area, per-storey areas, total floor area, aspect ratio |
| `functions` | which room types the plan covers |
| `furniture` | item count, distinct assets, parts/table densities, per-room-type asset sets, combo hash |
| `cameras` | per-room counts, pairwise spacing in **3D**, spacing within one storey, navigation distance through the room graph, unreachable pairs, rooms without a viewpoint |
| `lighting` | sun elevation/azimuth, sky strength, exposure, windows |

and then aggregates them: histograms, per-family counts, and **topology
repeats** -- scenes whose canonical structure fingerprint is identical.  Two
houses are the same topology when the same functions sit on the same storeys
with the same connections and cycle rank, no matter how the rooms are named, in
which order the lists were written, or which furniture stands in them: a chain,
a star and a ring of four rooms are three topologies, and a stair that joins two
storeys is part of the structure instead of a missing edge.  The fingerprint's
hash only groups *candidates*; membership in a group is decided by comparing the
fingerprints themselves, so different graphs that happen to share a hash stay
apart (`hash_collisions` counts how often that happened).  `plan_hash` is the
stricter neighbour (structure + coarse shape bins) for "near-identical plan"
detection.

Connections come from the spec contract: `connectors` **and** the legacy
`openings`, merged the same way the builder merges them (a spec that does not
parse falls back to a tolerant raw reading).  Areas are union areas of the room
outlines, sampled per compressed x-strip -- exact for the rectilinear polygons
the generator writes, a midpoint estimate for oblique ones -- so an L-shaped or
stacked plan no longer reports its bounding rectangle as its floor space.

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
a *new* structure fingerprint, filled up in seed order.
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

#: connection kinds the spec contract carries (see blender_agent/spec.py)
CONNECTION_KINDS = ("door", "open_passage", "stair", "ramp")
CROSS_FLOOR_KINDS = ("stair", "ramp")

#: world z of a camera that only carries an XY position (legacy specs)
DEFAULT_CAMERA_Z_M = 1.5

#: individualisation/refinement visits budget for one structure fingerprint
CANONICAL_BUDGET = 4096


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


def _as_list(value) -> list:
    """A JSON list, or nothing: one malformed block must not sink the report."""
    return value if isinstance(value, list) else []


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
        if not isinstance(spec, dict):
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


def _spec_module():
    """``blender_agent.spec`` when importable (it is pure python, no bpy)."""
    try:
        from blender_agent import spec as spec_mod  # noqa: PLC0415 - optional import
    except Exception:
        return None
    return spec_mod


# --------------------------------------------------------------------------- #
# the room graph of one scene
# --------------------------------------------------------------------------- #
def scene_graph(spec: dict) -> dict:
    """Rooms, storeys and *every* connection of one spec dict.

    The frozen contract in ``blender_agent.spec`` is preferred, so ``openings``
    and ``connectors`` end up in one graph exactly the way the builder merges
    them; a spec that does not parse falls back to a tolerant raw reading.

    Returns ``rooms`` (``id``/``type``/``floor`` index/``polygon``), ``floor_ids``
    and ``floor_elevations`` in spec order, ``connections`` as
    ``(from_room, to_room, kind)``, ``cameras`` with a 3D position, and which
    reader produced it.
    """
    module = _spec_module()
    if module is not None:
        try:
            parsed = module.parse_spec(spec)
            floor_ids = [str(floor.id) for floor in parsed.floors]
            floor_index = {floor_id: index for index, floor_id in enumerate(floor_ids)}
            rooms = [{"id": str(room.id), "type": str(room.type),
                      "floor": floor_index.get(str(room.floor_id), 0 if not floor_ids else -1),
                      "polygon": [[float(point[0]), float(point[1])] for point in room.polygon]}
                     for room in parsed.rooms]
            cameras = [{"name": str(camera.name), "room": str(camera.room),
                        "pos": [float(value) for value in camera.pos]}
                       for camera in parsed.cameras]
            return {
                "rooms": rooms,
                "floor_ids": floor_ids,
                "floor_elevations": [float(floor.elevation) for floor in parsed.floors],
                "connections": [(str(conn.from_room), str(conn.to_room), str(conn.kind))
                                for conn in parsed.connections()],
                "cameras": cameras,
                "source": "spec",
            }
        except Exception:
            pass
    return _raw_scene_graph(spec)


def _raw_scene_graph(spec: dict) -> dict:
    raw_floors = _as_list(spec.get("floors"))
    floor_ids = [str(floor.get("id")) for floor in raw_floors if isinstance(floor, dict)]
    floor_elevations = [float(floor.get("elevation", 0.0)) for floor in raw_floors
                        if isinstance(floor, dict)]
    floor_index = {floor_id: index for index, floor_id in enumerate(floor_ids)}
    rooms = []
    for room in _as_list(spec.get("rooms")):
        if not isinstance(room, dict):
            continue
        floor_id = room.get("floor_id")
        if floor_id is None:
            index = 0
        else:
            index = floor_index.setdefault(str(floor_id), len(floor_index))
        rooms.append({"id": str(room.get("id")), "type": str(room.get("type", "unknown")),
                      "floor": index,
                      "polygon": [[float(point[0]), float(point[1])]
                                  for point in _as_list(room.get("polygon"))
                                  if len(point) >= 2]})
    known = {room["id"] for room in rooms}
    connections: List[Tuple[str, str, str]] = []
    raw_connectors = spec.get("connectors")
    if raw_connectors is None:
        raw_connectors = spec.get("connector")
    for connector in _as_list(raw_connectors):
        if not isinstance(connector, dict):
            continue
        first, second = str(connector.get("from_room")), str(connector.get("to_room"))
        if first in known and second in known:
            connections.append((first, second, str(connector.get("kind", "door"))))
    for opening in _as_list(spec.get("openings")):
        if not isinstance(opening, dict):
            continue
        between = [str(value) for value in _as_list(opening.get("between"))]
        if len(between) == 2 and between[0] in known and between[1] in known:
            connections.append((between[0], between[1], str(opening.get("kind", "door"))))
    cameras = []
    for camera in _as_list(spec.get("cameras")):
        if not isinstance(camera, dict):
            continue
        cameras.append({"name": str(camera.get("name")), "room": str(camera.get("room")),
                        "pos": [float(value) for value in _as_list(camera.get("pos"))]})
    return {
        "rooms": rooms,
        "floor_ids": floor_ids,
        "floor_elevations": floor_elevations,
        "connections": connections,
        "cameras": cameras,
        "source": "raw",
    }


def floor_label(graph: dict, index: int) -> str:
    """Readable storey name: the declared floor id, else ``floor<index>``."""
    labels = graph["floor_ids"]
    if 0 <= index < len(labels):
        return str(labels[index])
    return "unassigned" if index < 0 else f"floor{index}"


def floor_elevation(graph: dict, index: int) -> float:
    elevations = graph["floor_elevations"]
    if 0 <= index < len(elevations):
        return elevations[index]
    return 0.0


# --------------------------------------------------------------------------- #
# areas: union of room outlines, not the bounding rectangle
# --------------------------------------------------------------------------- #
def _polygon_spans(polygon: Sequence[Sequence[float]], x: float) -> List[Tuple[float, float]]:
    """The y intervals a vertical line at ``x`` cuts out of one polygon."""
    crossings: List[float] = []
    count = len(polygon)
    for index in range(count):
        x1, y1 = float(polygon[index][0]), float(polygon[index][1])
        x2, y2 = float(polygon[(index + 1) % count][0]), float(polygon[(index + 1) % count][1])
        if (x1 <= x < x2) or (x2 <= x < x1):
            crossings.append(y1 + (y2 - y1) * (x - x1) / (x2 - x1))
    crossings.sort()
    return [(crossings[i], crossings[i + 1]) for i in range(0, len(crossings) - 1, 2)]


def _union_span_length(spans: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    end: Optional[float] = None
    for start, stop in sorted(spans):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def polygon_union_area(polygons: Sequence[Sequence[Sequence[float]]]) -> float:
    """Area of the union of the outlines.

    The x axis is split at every vertex coordinate and each strip contributes
    ``width * union of the y intervals at the strip centre``.  That is exact for
    rectilinear outlines (what the families and the duplex generator write) and
    a midpoint estimate for oblique ones; overlapping rooms are counted once.
    """
    shapes = [[[float(point[0]), float(point[1])] for point in polygon]
              for polygon in polygons if len(polygon) >= 3]
    if not shapes:
        return 0.0
    xs = sorted({point[0] for shape in shapes for point in shape})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right - left <= 1e-9:
            continue
        middle = 0.5 * (left + right)
        spans: List[Tuple[float, float]] = []
        for shape in shapes:
            spans.extend(_polygon_spans(shape, middle))
        area += (right - left) * _union_span_length(spans)
    return area


def _centroid(polygon: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """Area centroid of a simple polygon (falls back to the vertex mean)."""
    count = len(polygon)
    if count == 0:
        return 0.0, 0.0
    twice_area = 0.0
    cx = cy = 0.0
    for index in range(count):
        x1, y1 = float(polygon[index][0]), float(polygon[index][1])
        x2, y2 = float(polygon[(index + 1) % count][0]), float(polygon[(index + 1) % count][1])
        cross = x1 * y2 - x2 * y1
        twice_area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(twice_area) <= 1e-9:
        return (sum(float(point[0]) for point in polygon) / count,
                sum(float(point[1]) for point in polygon) / count)
    return cx / (3.0 * twice_area), cy / (3.0 * twice_area)


# --------------------------------------------------------------------------- #
# canonical fingerprint of the attributed room graph
# --------------------------------------------------------------------------- #
def _refine(node_count: int, adjacency: Sequence[Sequence[Tuple[int, str]]],
            colors: Sequence[int]) -> List[int]:
    """One canonical colour-refinement pass, repeated until the partition holds."""
    while True:
        signatures = []
        for node in range(node_count):
            neighbours = sorted((kind, colors[other]) for other, kind in adjacency[node])
            signatures.append((colors[node], tuple(neighbours)))
        unique = sorted(set(signatures))
        index = {signature: position for position, signature in enumerate(unique)}
        refined = [index[signature] for signature in signatures]
        if len(unique) == len(set(colors)):
            return refined
        colors = refined


def _certificate(node_count: int, adjacency: Sequence[Sequence[Tuple[int, str]]],
                 node_attrs: Sequence[tuple], colors: Sequence[int]) -> str:
    """Full graph serialisation in canonical (colour) order -- all colours distinct."""
    order = sorted(range(node_count), key=lambda node: (colors[node], node))
    rank = {node: position for position, node in enumerate(order)}
    payload = []
    for node in order:
        payload.append([list(node_attrs[node]),
                        sorted([kind, rank[other]] for other, kind in adjacency[node])])
    return json.dumps(payload, separators=(",", ":"))


def _quotient_key(node_count: int, adjacency: Sequence[Sequence[Tuple[int, str]]],
                  node_attrs: Sequence[tuple], colors: Sequence[int]) -> str:
    """Invariant but possibly incomplete key: the coloured quotient graph."""
    payload = sorted([list(node_attrs[node]),
                      sorted([kind, colors[other]] for other, kind in adjacency[node])]
                     for node in range(node_count))
    return json.dumps(payload, separators=(",", ":"))


def canonical_graph_key(node_attrs: Sequence[tuple],
                        edges: Sequence[Tuple[int, int, str]],
                        floor_count: int) -> Tuple[str, bool]:
    """Canonical fingerprint of the attributed room graph, and whether it is exact.

    ``node_attrs`` are per room instance (function, storey index), ``edges`` are
    ``(node, node, kind)`` with ``kind`` a connection type.  Colour refinement
    plus individualisation search gives a canonical form: it does not depend on
    room names, node or edge order, and a chain, a star and a ring of the same
    rooms produce different keys.  ``exact`` is false when the search budget ran
    out and the key fell back to the refined quotient (still order invariant,
    but two non-isomorphic graphs may then share it).
    """
    node_count = len(node_attrs)
    adjacency: List[List[Tuple[int, str]]] = [[] for _ in range(node_count)]
    for first, second, kind in edges:
        if first == second:
            continue
        adjacency[first].append((second, str(kind)))
        adjacency[second].append((first, str(kind)))
    attrs = [tuple(attr) for attr in node_attrs]
    distinct = sorted(set(attrs))
    initial = [distinct.index(attr) for attr in attrs]
    state = {"visits": 0, "exact": True}

    def explore(colors: List[int]) -> str:
        state["visits"] += 1
        if state["visits"] > CANONICAL_BUDGET:
            state["exact"] = False
            return _quotient_key(node_count, adjacency, attrs, colors)
        colors = _refine(node_count, adjacency, colors)
        classes: Dict[int, List[int]] = {}
        for node, color in enumerate(colors):
            classes.setdefault(color, []).append(node)
        ties = sorted(color for color, members in classes.items() if len(members) > 1)
        if not ties:
            return _certificate(node_count, adjacency, attrs, colors)
        fresh = max(colors) + 1
        best: Optional[str] = None
        for node in classes[ties[0]]:
            child = list(colors)
            child[node] = fresh
            key = explore(child)
            if best is None or key < best:
                best = key
        return best

    certificate = explore(initial)
    marker = "exact" if state["exact"] else "quotient"
    return f"floors={int(floor_count)}|{marker}|{certificate}", bool(state["exact"])


def _cycle_rank(node_count: int, edges: Sequence[Tuple[int, int, str]]) -> int:
    """``E - V + C`` of the room graph: independent loops of the plan."""
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first, second, _ in edges:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_a] = root_b
    if node_count == 0:
        return 0
    components = len({find(node) for node in range(node_count)})
    return max(0, len(edges) - node_count + components)


# --------------------------------------------------------------------------- #
# per-scene metrics
# --------------------------------------------------------------------------- #
def _camera_metrics(graph: dict, rooms: Sequence[dict], room_by_id: Dict[str, dict],
                    edges: Sequence[Tuple[str, str, str]]) -> dict:
    """Per-room counts plus 3D, same-storey and navigation camera distances."""
    cameras = graph["cameras"]
    per_room: Dict[str, int] = {}
    positions: List[Tuple[float, float, float]] = []
    camera_rooms: List[str] = []
    floors: List[int] = []
    for camera in cameras:
        room_id = camera["room"]
        per_room[room_id] = per_room.get(room_id, 0) + 1
        room = room_by_id.get(room_id)
        position = list(camera["pos"])
        if len(position) >= 3:
            z = float(position[2])
        else:
            z = floor_elevation(graph, room["floor"] if room else 0) + DEFAULT_CAMERA_Z_M
        x = float(position[0]) if position else 0.0
        y = float(position[1]) if len(position) > 1 else 0.0
        positions.append((x, y, z))
        camera_rooms.append(room_id)
        floors.append(room["floor"] if room else 0)

    gaps_3d: List[float] = []
    gaps_same_floor: List[float] = []
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            gap = math.dist(positions[first], positions[second])
            gaps_3d.append(gap)
            if floors[first] == floors[second]:
                gaps_same_floor.append(gap)

    navigation = _navigation_distances(graph, rooms, room_by_id, edges, positions, camera_rooms)

    return {
        "count": len(cameras),
        "per_room": per_room,
        "floor_counts": {floor_label(graph, floor): sum(1 for value in floors if value == floor)
                         for floor in sorted(set(floors))},
        "rooms_without_camera": sorted(room["id"] for room in rooms
                                       if room["id"] not in per_room
                                       and room.get("type") not in TRANSIT_TYPES),
        "min_gap_3d_m": round(min(gaps_3d), 3) if gaps_3d else None,
        "mean_gap_3d_m": round(statistics.fmean(gaps_3d), 3) if gaps_3d else None,
        "min_gap_same_floor_m": round(min(gaps_same_floor), 3) if gaps_same_floor else None,
        "mean_gap_same_floor_m": round(statistics.fmean(gaps_same_floor), 3)
        if gaps_same_floor else None,
        "min_gap_nav_m": round(min(navigation["gaps"]), 3) if navigation["gaps"] else None,
        "mean_gap_nav_m": round(statistics.fmean(navigation["gaps"]), 3)
        if navigation["gaps"] else None,
        "unreachable_pairs": navigation["unreachable"],
    }


def _navigation_distances(graph: dict, rooms: Sequence[dict], room_by_id: Dict[str, dict],
                          edges: Sequence[Tuple[str, str, str]],
                          positions: Sequence[Tuple[float, float, float]],
                          camera_rooms: Sequence[str]) -> dict:
    """Camera-to-camera distance *through the plan*, via the room graph.

    A camera pair in one room is the direct 3D distance (no wall between them);
    otherwise the walk is camera -> its room centre -> shortest connection path
    -> room centre -> camera.  That is a lower bound on the real walking
    distance (it ignores furniture and where the door sits on the wall) but it
    never reports two storeys as zero metres apart, which an XY-only spacing did.
    """
    centroids: Dict[str, Tuple[float, float, float]] = {}
    for room in rooms:
        x, y = _centroid(room["polygon"])
        centroids[room["id"]] = (x, y, floor_elevation(graph, room["floor"]))
    weights: Dict[str, Dict[str, float]] = {}
    for first, second, _kind in edges:
        if first == second or first not in centroids or second not in centroids:
            continue
        span = math.dist(centroids[first], centroids[second])
        for node, other in ((first, second), (second, first)):
            best = weights.setdefault(node, {}).get(other)
            if best is None or span < best:
                weights[node][other] = span

    def shortest(source: str) -> Dict[str, float]:
        distances: Dict[str, float] = {source: 0.0}
        pending = {source}
        while pending:
            node = min(pending, key=lambda value: distances[value])
            pending.discard(node)
            for neighbour, weight in (weights.get(node) or {}).items():
                candidate = distances[node] + weight
                if candidate < distances.get(neighbour, float("inf")):
                    distances[neighbour] = candidate
                    pending.add(neighbour)
        return distances

    gaps: List[float] = []
    unreachable = 0
    paths: Dict[str, Dict[str, float]] = {}
    for first in range(len(positions)):
        room_first = camera_rooms[first]
        for second in range(first + 1, len(positions)):
            room_second = camera_rooms[second]
            if room_first not in room_by_id or room_second not in room_by_id:
                unreachable += 1
                continue
            if room_first == room_second:
                gaps.append(math.dist(positions[first], positions[second]))
                continue
            if room_first not in paths:
                paths[room_first] = shortest(room_first)
            through = paths[room_first].get(room_second)
            if through is None:
                unreachable += 1
                continue
            gaps.append(math.dist(positions[first], centroids[room_first]) + through
                        + math.dist(centroids[room_second], positions[second]))
    return {"gaps": gaps, "unreachable": unreachable}


def scene_metrics(spec: dict, seed: Optional[int] = None, layout_report: Optional[dict] = None,
                  path: Optional[str] = None) -> dict:
    """Structure, shape, function, furniture, camera and lighting numbers."""
    graph = scene_graph(spec)
    rooms = graph["rooms"]
    room_by_id = {room["id"]: room for room in rooms}
    index_by_id = {room["id"]: index for index, room in enumerate(rooms)}
    floor_count = len(graph["floor_ids"]) or 1

    edges: List[Tuple[str, str, str]] = []
    for first, second, kind in graph["connections"]:
        if first not in index_by_id or second not in index_by_id or first == second:
            continue
        low, high = sorted((first, second))
        edges.append((low, high, str(kind)))
    edges = sorted(set(edges))
    graph_edges = [(index_by_id[first], index_by_id[second], kind) for first, second, kind in edges]

    kinds: Dict[str, int] = {}
    for _first, _second, kind in edges:
        kinds[kind] = kinds.get(kind, 0) + 1
    cross_floor = sum(1 for first, second, _kind in edges
                      if room_by_id[first]["floor"] != room_by_id[second]["floor"])

    node_attrs = [(room["type"], room["floor"]) for room in rooms]
    structure_key, structure_exact = canonical_graph_key(node_attrs, graph_edges, floor_count)
    cycles = _cycle_rank(len(rooms), graph_edges)

    type_counts: Dict[str, int] = {}
    for room in rooms:
        type_counts[room["type"]] = type_counts.get(room["type"], 0) + 1

    polygons = [room["polygon"] for room in rooms if len(room["polygon"]) >= 3]
    points = [point for polygon in polygons for point in polygon]
    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width, depth = max(xs) - min(xs), max(ys) - min(ys)
        bbox_area = width * depth
        aspect = round(max(width, depth) / max(1e-6, min(width, depth)), 2)
    else:
        width = depth = bbox_area = 0.0
        aspect = 1.0
    footprint = polygon_union_area(polygons)
    floor_areas = {
        floor_label(graph, floor): round(polygon_union_area(
            [room["polygon"] for room in rooms if room["floor"] == floor]), 2)
        for floor in sorted({room["floor"] for room in rooms})
    }
    total_floor_area = sum(floor_areas.values())

    furniture = [item for item in _as_list(spec.get("furniture")) if isinstance(item, dict)]
    furniture_by_room: Dict[str, List[str]] = {}
    for item in furniture:
        furniture_by_room.setdefault(item.get("room", "?"), []).append(str(item.get("asset")))
    combo: Dict[str, List[str]] = {}
    for room in rooms:
        assets = sorted(furniture_by_room.get(room["id"], []))
        combo.setdefault(room["type"], []).extend(assets)
    combo = {room_type_name: sorted(assets) for room_type_name, assets in combo.items()}

    style = spec.get("style") if isinstance(spec.get("style"), dict) else {}
    structure = {
        "floors": floor_count,
        "rooms": type_counts,
        "nodes": [{"type": room["type"], "floor": floor_label(graph, room["floor"])}
                  for room in sorted(rooms, key=lambda room: (room["floor"], room["type"],
                                                              room["id"]))],
        "edges": [{"rooms": [first, second],
                   "types": [room_by_id[first]["type"], room_by_id[second]["type"]],
                   "kind": kind} for first, second, kind in edges],
        "edge_kinds": dict(sorted(kinds.items())),
        "cycles": cycles,
        "exact": structure_exact,
    }
    plan_key = (f"{structure_key}|aspect={_round(aspect, 0.5)}"
                f"|bbox={_round(bbox_area, 10.0)}|floor_area={_round(total_floor_area, 10.0)}")

    return {
        "seed": seed,
        "path": path,
        "family": (layout_report or {}).get("family") or (layout_report or {}).get("shape"),
        "rooms": len(rooms),
        "room_types": type_counts,
        "floors": floor_count,
        "floor_labels": list(graph["floor_ids"]),
        "connections": {"count": len(edges), "kinds": dict(sorted(kinds.items())),
                        "cross_floor": cross_floor},
        "structure": structure,
        "structure_key": structure_key,
        "structure_hash": _hash(structure_key),
        "plan_key": plan_key,
        "plan_hash": _hash(plan_key),
        "shape": {"width_m": round(width, 2), "depth_m": round(depth, 2),
                  "bbox_area_m2": round(bbox_area, 2), "aspect": aspect,
                  "footprint_area_m2": round(footprint, 2),
                  "total_floor_area_m2": round(total_floor_area, 2),
                  "floor_areas_m2": floor_areas,
                  "footprint_ratio": round(footprint / bbox_area, 3) if bbox_area > 0 else None},
        "functions": sorted(type_counts),
        "furniture": {"count": len(furniture),
                      "distinct_assets": len({item.get("asset") for item in furniture}),
                      "combo_hash": _hash(combo), "per_room_type": combo},
        "cameras": _camera_metrics(graph, rooms, room_by_id, edges),
        "lighting": {"sun_elevation_deg": style.get("sun_elevation_deg"),
                     "sun_azimuth_deg": style.get("sun_azimuth_deg"),
                     "sky_strength": style.get("sky_strength"),
                     "exposure": style.get("exposure"),
                     "windows": len(_as_list(spec.get("windows")))},
    }


def _histogram(values: Iterable[float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda entry: float(entry[0])))


def _exact_groups(metrics: Sequence[dict], key_field: str) -> Tuple[Dict[str, List[dict]], int]:
    """Group by the canonical key: hash buckets first, exact key comparison after."""
    buckets: Dict[str, List[dict]] = {}
    for entry in metrics:
        buckets.setdefault(_hash(entry[key_field]), []).append(entry)
    groups: Dict[str, List[dict]] = {}
    collisions = 0
    for bucket in buckets.values():
        split: Dict[str, List[dict]] = {}
        for entry in bucket:
            split.setdefault(entry[key_field], []).append(entry)
        if len(split) > 1:
            collisions += 1
        groups.update(split)
    return groups, collisions


def _scene_min(metrics: Sequence[dict], path: Sequence[str]) -> Optional[float]:
    values = []
    for entry in metrics:
        node = entry
        for step in path:
            node = node[step]
        if node is not None:
            values.append(node)
    return round(min(values), 3) if values else None


def _scene_mean(metrics: Sequence[dict], path: Sequence[str]) -> Optional[float]:
    values = []
    for entry in metrics:
        node = entry
        for step in path:
            node = node[step]
        if node is not None:
            values.append(node)
    return round(statistics.fmean(values), 3) if values else None


def diversity_summary(metrics: Sequence[dict]) -> dict:
    """Aggregate per-scene metrics: histograms, coverage and topology repeats."""
    if not metrics:
        return {"count": 0}

    types: Dict[str, int] = {}
    families: Dict[str, int] = {}
    kind_totals: Dict[str, int] = {}
    for entry in metrics:
        families[str(entry.get("family"))] = families.get(str(entry.get("family")), 0) + 1
        for name in entry["functions"]:
            types[name] = types.get(name, 0) + 1
        for kind, count in entry["connections"]["kinds"].items():
            kind_totals[kind] = kind_totals.get(kind, 0) + count

    groups, structure_collisions = _exact_groups(metrics, "structure_key")
    plan_groups, plan_collisions = _exact_groups(metrics, "plan_key")

    def repeat_report(bucket: Dict[str, List[dict]], collisions: int) -> dict:
        repeated = {key: entries for key, entries in bucket.items() if len(entries) > 1}
        top = sorted(repeated.items(), key=lambda item: -len(item[1]))[:5]
        return {
            "unique": len(bucket),
            "duplicate_rate": round(1.0 - len(bucket) / float(len(metrics)), 4),
            "hash_collisions": collisions,
            "groups": [
                {"hash": _hash(key),
                 "key": key,
                 "count": len(entries),
                 "seeds": sorted([entry["seed"] for entry in entries
                                  if entry.get("seed") is not None])[:20],
                 "families": sorted({str(entry.get("family")) for entry in entries}),
                 "furniture_combos": len({entry["furniture"]["combo_hash"] for entry in entries}),
                 "note": ("same room graph, different furniture"
                          if len({entry["furniture"]["combo_hash"] for entry in entries}) > 1
                          else "same room graph and the same furniture sets")}
                for key, entries in top
            ],
            "repeat_scenes": sum(len(entries) for entries in repeated.values()),
        }

    furniture_metrics = [entry["furniture"] for entry in metrics]
    camera_counts = [entry["cameras"]["count"] for entry in metrics]
    room_counts = [entry["rooms"] for entry in metrics]
    aspects = [entry["shape"]["aspect"] for entry in metrics]
    bbox_areas = [entry["shape"]["bbox_area_m2"] for entry in metrics]
    footprints = [entry["shape"]["footprint_area_m2"] for entry in metrics]
    floor_areas = [entry["shape"]["total_floor_area_m2"] for entry in metrics]
    sun_azimuths = [entry["lighting"]["sun_azimuth_deg"] for entry in metrics
                    if entry["lighting"]["sun_azimuth_deg"] is not None]
    cycles = [entry["structure"]["cycles"] for entry in metrics]
    return {
        "count": len(metrics),
        "families": dict(sorted(families.items())),
        "rooms": {"histogram": _histogram(room_counts), "min": min(room_counts),
                  "max": max(room_counts), "mean": round(statistics.fmean(room_counts), 2)},
        "floors": _histogram([entry["floors"] for entry in metrics]),
        "cycles": {"histogram": _histogram(cycles), "max": max(cycles),
                   "loops": sum(1 for value in cycles if value > 0)},
        "connections": {
            "total": sum(entry["connections"]["count"] for entry in metrics),
            "kinds": dict(sorted(kind_totals.items())),
            "cross_floor": sum(entry["connections"]["cross_floor"] for entry in metrics),
            "scenes_with_stairs": sum(1 for entry in metrics
                                      if entry["connections"]["kinds"].get("stair")),
            "scenes_with_ramps": sum(1 for entry in metrics
                                     if entry["connections"]["kinds"].get("ramp")),
            "scenes_without_connection": sum(1 for entry in metrics
                                             if not entry["connections"]["count"]),
        },
        "function_coverage": dict(sorted(types.items(), key=lambda entry: -entry[1])),
        "shape": {"aspect_min": min(aspects), "aspect_max": max(aspects),
                  "aspect_mean": round(statistics.fmean(aspects), 2),
                  "bbox_area_min_m2": min(bbox_areas), "bbox_area_max_m2": max(bbox_areas),
                  "bbox_area_mean_m2": round(statistics.fmean(bbox_areas), 2),
                  "footprint_area_min_m2": min(footprints),
                  "footprint_area_max_m2": max(footprints),
                  "footprint_area_mean_m2": round(statistics.fmean(footprints), 2),
                  "total_floor_area_min_m2": min(floor_areas),
                  "total_floor_area_max_m2": max(floor_areas),
                  "total_floor_area_mean_m2": round(statistics.fmean(floor_areas), 2)},
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
                    "min_gap_3d_m_min": _scene_min(metrics, ("cameras", "min_gap_3d_m")),
                    "mean_gap_3d_m_mean": _scene_mean(metrics, ("cameras", "mean_gap_3d_m")),
                    "min_gap_same_floor_m_min": _scene_min(
                        metrics, ("cameras", "min_gap_same_floor_m")),
                    "mean_gap_same_floor_m_mean": _scene_mean(
                        metrics, ("cameras", "mean_gap_same_floor_m")),
                    "min_gap_nav_m_min": _scene_min(metrics, ("cameras", "min_gap_nav_m")),
                    "mean_gap_nav_m_mean": _scene_mean(metrics, ("cameras", "mean_gap_nav_m")),
                    "unreachable_pairs": sum(entry["cameras"]["unreachable_pairs"]
                                             for entry in metrics)},
        "lighting": {"sun_azimuth_distinct": len(set(sun_azimuths)),
                     "sun_azimuth_values": sorted(set(sun_azimuths))[:12]},
        "structures": repeat_report(groups, structure_collisions),
        "plans": repeat_report(plan_groups, plan_collisions),
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

    seen_structures = {entry["structure_key"] for entry in chosen}
    for entry in sorted(metrics, key=lambda item: item.get("seed") if item.get("seed") is not None else -1):
        if len(chosen) >= size:
            break
        if entry in chosen or entry["structure_key"] in seen_structures:
            continue
        seen_structures.add(entry["structure_key"])
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
    print(f"floors (storeys)  : {summary.get('floors')}")
    print(f"connections       : {summary.get('connections')}")
    print(f"cycles            : {summary.get('cycles')}")
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
